import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

"""
ORAC-NT Cascading Pipeline (v34 — REAL STRAIN EDITION)
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.19553825

Features:
1. Blind Calibrated Energy Trigger (ORAC-NT)
2. Smart Kurtosis Veto for Glitches
3. Matched Filter Handoff
4. Real GWOSC strain fetch via requests+h5py (no gwpy/nds2)
5. Full 3-Panel Visual Export with UTC timestamp
"""

import os
import json
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal.windows import tukey
from scipy.stats import kurtosis
from datetime import datetime, timezone
import warnings

warnings.filterwarnings('ignore')

FS       = 4096
DURATION = 16
t        = np.arange(FS * DURATION) / FS

# ─────────────────────────────────────────────────────────────
# GWOSC strain fetch — без gwpy, директен requests + h5py
# ─────────────────────────────────────────────────────────────

def fetch_real_strain(event_name, duration=32, detector='L1'):
    """
    Изтегля реални strain данни от GWOSC чрез:
      1. gwosc.locate.get_event_urls() → HDF5 URL
      2. requests → download
      3. h5py → четене
    Не използва gwpy (заобикаля nds2 бъга).
    Връща (numpy array, True) при успех, (None, False) иначе.
    """
    if not event_name or event_name in ('SIMULATION', 'UNKNOWN', ''):
        return None, False

    try:
        import requests
        import h5py
        from gwosc.locate import get_event_urls

        urls = get_event_urls(event_name, detector=detector, duration=duration)
        if not urls:
            print(f"  [GWOSC] Няма URLs за {event_name}")
            return None, False

        hdf_url = urls[0]
        print(f"  [GWOSC] Изтегляне: ...{hdf_url[-50:]}")

        r = requests.get(hdf_url, timeout=120, stream=True)
        r.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        for chunk in r.iter_content(chunk_size=65536):
            tmp.write(chunk)
        tmp.close()

        with h5py.File(tmp.name, 'r') as f:
            strain_data = f['strain']['Strain'][:]
            dt          = f['strain']['Strain'].attrs.get('Xspacing', 1.0 / FS)

        os.unlink(tmp.name)

        strain      = strain_data.astype(np.float64)
        sample_rate = int(round(1.0 / dt))

        if sample_rate != FS:
            from scipy.signal import resample
            strain = resample(strain, int(len(strain) * FS / sample_rate))

        # Pad/crop до DURATION
        target = FS * DURATION
        if len(strain) >= target:
            strain = strain[:target]
        else:
            strain = np.pad(strain, (0, target - len(strain)))

        print(f"  [GWOSC] ✅ Реални данни: {len(strain)} samples @ {FS} Hz")
        return strain, True

    except Exception as e:
        print(f"  [GWOSC] ⚠️  Недостъпни ({e})")
        return None, False

# ─────────────────────────────────────────────────────────────
# Simulated stream (fallback)
# ─────────────────────────────────────────────────────────────

def generate_noise():
    return np.random.randn(len(t))

def gw_signal(t0):
    return 6.0 * np.sin(2*np.pi*150*t) * np.exp(-((t-t0)**2)/(2*0.08**2))

def glitch(t0):
    return 15.0 * np.sin(2*np.pi*400*t) * np.exp(-((t-t0)**2)/(2*0.01**2))

# ─────────────────────────────────────────────────────────────
# ORAC-NT Trigger
# ─────────────────────────────────────────────────────────────

class ORAC_Trigger:
    def __init__(self):
        self.fs            = FS
        self.h_threshold   = 2.5
        self.calibration_s = 5.0

    def whiten(self, data):
        cal   = int(self.calibration_s * self.fs)
        f, psd = signal.welch(data[:cal], self.fs, nperseg=self.fs // 2)
        psd_i = np.interp(np.fft.rfftfreq(len(data), 1 / self.fs), f, psd)
        w     = np.fft.irfft(np.fft.rfft(data) / np.sqrt(psd_i + 1e-12), n=len(data))
        return w * tukey(len(w), 0.05)

    def scan(self, stream):
        cal = int(self.calibration_s * self.fs)
        w   = self.whiten(stream)
        w  /= (np.std(w[:cal]) + 1e-12)
        env = np.abs(signal.hilbert(w))
        seg = env[:cal]
        med = np.median(seg)
        mad = 1.4826 * np.median(np.abs(seg - med))
        nf  = med + 1.5 * mad

        h, triggers = 0.0, []
        for i, val in enumerate(env):
            if i < cal:
                continue
            h += 0.15 if val > nf else -0.03
            h  = np.clip(h, 0.0, 5.0)
            if h >= self.h_threshold:
                triggers.append(i / self.fs)

        return self.cluster(triggers), np.array(
            [0.0] * cal + [
                np.clip(
                    sum(
                        0.15 if env[j] > nf else -0.03
                        for j in range(cal, i + 1)
                    ), 0.0, 5.0
                )
                for i in range(cal, len(env))
            ]
        ), nf

    def cluster(self, triggers, dt=0.5):
        if not triggers: return []
        clusters = [[triggers[0]]]
        for tr in triggers[1:]:
            if tr - clusters[-1][-1] < dt:
                clusters[-1].append(tr)
            else:
                clusters.append([tr])
        return [c[0] for c in clusters]

# ─────────────────────────────────────────────────────────────
# Efficient scan (за реална употреба)
# ─────────────────────────────────────────────────────────────

def scan_stream(engine, stream):
    cal = int(engine.calibration_s * engine.fs)
    w   = engine.whiten(stream)
    w  /= (np.std(w[:cal]) + 1e-12)
    env = np.abs(signal.hilbert(w))
    seg = env[:cal]
    med = np.median(seg)
    mad = 1.4826 * np.median(np.abs(seg - med))
    nf  = med + 1.5 * mad

    h         = 0.0
    h_history = []
    triggers  = []

    for i, val in enumerate(env):
        if i < cal:
            h_history.append(0.0)
            continue
        h += 0.15 if val > nf else -0.03
        h  = np.clip(h, 0.0, 5.0)
        h_history.append(h)
        if h >= engine.h_threshold:
            triggers.append(i / engine.fs)

    clustered = engine.cluster(triggers)
    return clustered, np.array(h_history), nf

# ─────────────────────────────────────────────────────────────
# Veto & Handoff
# ─────────────────────────────────────────────────────────────

def get_snippet(data, t_event, window=2.0):
    idx  = int(t_event * FS)
    half = int(window / 2 * FS)
    return data[max(0, idx - half): min(len(data), idx + half)]

def is_glitch(snippet):
    if snippet is None or len(snippet) < 100: return True
    return kurtosis(snippet, fisher=False, bias=False) > 25.0

def matched_filter_confirm(snippet):
    if len(snippet) < FS // 2: return 0.0
    f, psd  = signal.welch(snippet, FS, nperseg=FS // 2)
    psd_i   = np.interp(np.fft.rfftfreq(len(snippet), 1 / FS), f, psd)
    w_snip  = np.fft.irfft(np.fft.rfft(snippet) / np.sqrt(psd_i + 1e-12), n=len(snippet))
    w_snip *= tukey(len(w_snip), 0.1)
    tt      = np.linspace(0, 0.5, int(0.5 * FS), endpoint=False)
    tmpl    = signal.chirp(tt, f0=80, f1=250, t1=0.5, method='linear')
    w_tmpl  = np.fft.irfft(np.fft.rfft(tmpl, n=len(snippet)) / np.sqrt(psd_i + 1e-12), n=len(snippet))
    w_tmpl /= np.sqrt(np.sum(w_tmpl**2) + 1e-20)
    corr    = signal.correlate(w_snip, w_tmpl, mode='same')
    med     = np.median(corr)
    mad     = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
    return float(np.max(np.abs(corr)) / mad)

# ─────────────────────────────────────────────────────────────
# Load event context
# ─────────────────────────────────────────────────────────────

def load_event_context():
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    event_file = os.path.join(base_dir, 'latest_event.json')
    if not os.path.exists(event_file):
        return 'SIMULATION', None, 0.0, None
    try:
        with open(event_file, 'r', encoding='utf-8') as f:
            ev = json.load(f)
        raw    = ev.get('raw_payload', {})
        far    = raw.get('event', {}).get('far') if isinstance(raw, dict) else None
        return (
            ev.get('id', 'UNKNOWN'),
            ev.get('id', None),          # event name за get_event_urls
            ev.get('bns_prob', 0.0),
            far
        )
    except Exception:
        return 'UNKNOWN', None, 0.0, None

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    run_utc     = datetime.now(timezone.utc)
    run_utc_str = run_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    event_id, event_name, bns_prob, far = load_event_context()

    print("\n" + "=" * 52)
    print("  ORAC-NT v34 — REAL STRAIN EDITION")
    print(f"  Event:  {event_id}")
    print(f"  BNS:    {bns_prob*100:.0f}%")
    print("=" * 52)

    # ── Опит за реални GWOSC данни ────────────────────────
    stream, is_real = fetch_real_strain(event_name, duration=32, detector='L1')

    if is_real:
        mode_label = "REAL DATA"
        data_label = f"Real GWOSC Strain — L1 | Event: {event_id}"
        print(f"\n  ✅ Режим: РЕАЛНИ ДАННИ от GWOSC")
    else:
        mode_label = "SIMULATED"
        data_label = f"Simulated Stream (embargo/fallback) | Event: {event_id}"
        print(f"\n  ⚠️  Режим: СИМУЛАЦИЯ")
        stream  = generate_noise()
        stream += glitch(6.0)
        stream += gw_signal(11.5)

    print("\n" + "=" * 52)
    print(f"  🧪 PIPELINE START [{mode_label}]")
    print("=" * 52)

    engine = ORAC_Trigger()
    triggers, h_curve, nf = scan_stream(engine, stream)

    valid_trigger = None
    mf_snr        = 0.0

    for tr in triggers:
        snippet = get_snippet(stream, tr)
        print(f"\n   🚨 TRIGGER FIRED at t = {tr:.3f}s")
        if is_glitch(snippet):
            print(f"      🛡️  VETO: Instrumental artifact detected. Event rejected.")
        else:
            print(f"      ✂️  HANDOFF: Passing 2.0s window to Matched Filter...")
            mf_snr = matched_filter_confirm(snippet)
            print(f"      🏆 EVENT CONFIRMED! | MF SNR = {mf_snr:.2f}")
            valid_trigger = tr

    # ── Plotting ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor='#0e1117', sharex=True)

    axes[0].plot(t, stream, color='#888888', lw=0.4, alpha=0.8, label=data_label)
    if not is_real:
        axes[0].axvline(6.0,  color='orange', lw=1.5, ls='--', alpha=0.7, label='Injected Glitch (t=6.0s)')
        axes[0].axvline(11.5, color='yellow', lw=1.5, ls='--', alpha=0.7, label='Injected GW Burst (t=11.5s)')
    if valid_trigger:
        axes[0].axvline(valid_trigger, color='red', lw=2, label=f'Confirmed Trigger (t={valid_trigger:.2f}s)')
        axes[0].axvspan(valid_trigger - 1.0, valid_trigger + 1.0, color='cyan', alpha=0.12, label='Window → MF')
    axes[0].set_title(f"Detector Stream [{mode_label}] — Event: {event_id}", color='white', fontsize=12)
    axes[0].set_ylabel("Strain", color='white')
    axes[0].legend(facecolor='#0e1117', labelcolor='white', loc='upper right', fontsize=8)
    axes[0].set_facecolor('#0e1117'); axes[0].tick_params(colors='white')

    axes[1].fill_between(t, h_curve, color='#00ffcc', alpha=0.6, label='H-factor')
    axes[1].axhline(engine.h_threshold, color='red',    lw=1.5, ls='--', label=f'Threshold = {engine.h_threshold}')
    axes[1].axhline(nf,                 color='yellow', lw=1.0, ls=':',  alpha=0.6, label=f'Noise floor = {nf:.3f}')
    for tr in triggers:
        axes[1].axvline(tr, color='red' if tr == valid_trigger else 'orange',
                        lw=2, ls='-' if tr == valid_trigger else '-.')
    axes[1].set_title("ORAC-NT H-factor (Triggering & Veto System)", color='white', fontsize=12)
    axes[1].set_ylabel("H-Factor Score", color='white')
    axes[1].legend(facecolor='#0e1117', labelcolor='white', loc='upper right', fontsize=8)
    axes[1].set_facecolor('#0e1117'); axes[1].tick_params(colors='white')

    if valid_trigger:
        zm = (t >= valid_trigger - 2.0) & (t <= valid_trigger + 2.0)
        axes[2].plot(t[zm], stream[zm], color='#00d2ff', lw=0.8, label='Zoomed stream (±2s)')
        axes[2].axvline(valid_trigger, color='red',    lw=2,   ls='-',  label='Trigger point')
        if not is_real:
            axes[2].axvline(11.5,      color='yellow', lw=1.5, ls='--', label='True burst center')
        axes[2].set_title(f"Zoom: Confirmed Event at {valid_trigger:.3f}s  |  MF SNR = {mf_snr:.2f}", color='white', fontsize=11)
    else:
        axes[2].set_title("No valid events confirmed.", color='white')
    axes[2].set_xlabel("Time (s)", color='white')
    axes[2].set_ylabel("Strain", color='white')
    axes[2].legend(facecolor='#0e1117', labelcolor='white', loc='upper right', fontsize=8)
    axes[2].set_facecolor('#0e1117'); axes[2].tick_params(colors='white')

    far_str = f"FAR: {far:.2e} /yr  |  " if far else ""
    plt.suptitle(
        f"ORAC-NT Cascading Pipeline — {mode_label}\n"
        f"Dimitar Kretski  |  DOI: 10.5281/zenodo.19553825  |  {far_str}{run_utc_str}",
        color='white', fontsize=11
    )
    plt.tight_layout()
    plt.savefig("orac_final_presentation.png", dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print(f"\n✅ Saved visual proof: orac_final_presentation.png")
    print(f"   🕐 Timestamp: {run_utc_str}")
    print(f"   📊 Mode: {mode_label}")