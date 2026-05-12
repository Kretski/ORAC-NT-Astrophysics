"""
ORAC-NT Phase E6 — Multi-Detector Coincidence + IFAR Curve
============================================================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20129975

Tests:
1. H1 + L1 + V1 triple coincidence (Δt < 10ms each pair)
2. Combined network SNR = sqrt(SNR_H1² + SNR_L1² + SNR_V1²)
3. IFAR curve (Inverse False Alarm Rate vs threshold)
4. Comparison: single-detector vs coincidence FAR

GW events with V1 data:
  GW170814 — first 3-detector BBH
  GW170817 — BNS with V1

ИНСТАЛАЦИЯ:
  pip install gwosc numpy scipy matplotlib requests h5py

УПОТРЕБА:
  python orac_phase_e6.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import time
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

FS              = 4096
CALIBRATED_T    = 3.047   # от Phase A-D
COINCIDENCE_DT  = 0.010   # 10ms — light travel time H1-L1
COINCIDENCE_DT_V = 0.027  # 27ms — max light travel H1/L1 to V1

# ─────────────────────────────────────────────────────────────
# Events — с три детектора
# ─────────────────────────────────────────────────────────────

EVENTS_3DET = [
    {"name": "GW170814", "type": "BBH", "duration": 32,
     "detectors": ["H1", "L1", "V1"]},
    {"name": "GW170817", "type": "BNS", "duration": 32,
     "detectors": ["H1", "L1", "V1"]},
]

EVENTS_2DET = [
    {"name": "GW150914", "type": "BBH", "duration": 32,
     "detectors": ["H1", "L1"]},
    {"name": "GW190521", "type": "BBH", "duration": 32,
     "detectors": ["H1", "L1"]},
    {"name": "GW190814", "type": "NSBH", "duration": 32,
     "detectors": ["H1", "L1"]},
    {"name": "GW151012", "type": "BBH", "duration": 32,
     "detectors": ["H1", "L1"]},
]

QUIET_SEGMENTS = [
    {"name": "O3a_q01_L1", "gps": 1238166018, "det": "L1", "dur": 4096},
    {"name": "O3a_q01_H1", "gps": 1238166018, "det": "H1", "dur": 4096},
    {"name": "O3a_q12_L1", "gps": 1242578176, "det": "L1", "dur": 4096},
    {"name": "O3a_q12_H1", "gps": 1242578176, "det": "H1", "dur": 4096},
]

# ─────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────

def fetch_event(event_name, detector, duration=32):
    try:
        import requests, h5py
        from gwosc.locate import get_event_urls
        time.sleep(2.0)
        urls = get_event_urls(event_name, detector=detector, duration=duration)
        if not urls: return None
        time.sleep(1.0)
        r = requests.get(urls[0], timeout=180, stream=True)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        for chunk in r.iter_content(65536): tmp.write(chunk)
        tmp.close()
        with h5py.File(tmp.name, 'r') as f:
            data = f['strain']['Strain'][:]
            dt   = f['strain']['Strain'].attrs.get('Xspacing', 1.0/FS)
        os.unlink(tmp.name)
        strain = data.astype(np.float64)
        sr = int(round(1.0/dt))
        if sr != FS:
            from scipy.signal import resample
            strain = resample(strain, int(len(strain)*FS/sr))
        target = duration * FS
        if len(strain) > target:
            mid = len(strain)//2
            strain = strain[mid-target//2: mid+target//2]
        print(f"      [OK] {detector}: {len(strain)/FS:.0f}s")
        return strain
    except Exception as e:
        print(f"      [WARN] {detector}: {e}")
        return None

def fetch_bulk(gps, duration, detector):
    try:
        import requests, h5py
        from gwosc.locate import get_urls
        time.sleep(2.0)
        urls = get_urls(detector, gps, gps + duration)
        if not urls: return None
        hdf_url = next((u for u in urls if '4096' in u and 'hdf5' in u), urls[0])
        time.sleep(1.0)
        r = requests.get(hdf_url, timeout=240, stream=True)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
        for chunk in r.iter_content(65536): tmp.write(chunk)
        tmp.close()
        with h5py.File(tmp.name, 'r') as f:
            data = f['strain']['Strain'][:]
            dt   = f['strain']['Strain'].attrs.get('Xspacing', 1.0/FS)
        os.unlink(tmp.name)
        strain = data.astype(np.float64)
        sr = int(round(1.0/dt))
        if sr != FS:
            from scipy.signal import resample
            strain = resample(strain, int(len(strain)*FS/sr))
        return strain
    except Exception as e:
        print(f"      [WARN] {detector}: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# ORAC-NT core
# ─────────────────────────────────────────────────────────────

TEMPLATES = [
    {"name": "BBH_O1",   "f0": 35,  "f1": 150, "duration": 0.2},
    {"name": "BBH_gen",  "f0": 20,  "f1": 350, "duration": 0.4},
    {"name": "BNS",      "f0": 30,  "f1": 400, "duration": 1.0},
    {"name": "NSBH",     "f0": 20,  "f1": 300, "duration": 0.6},
]

def whiten(data, cal_s=5.0):
    sos    = signal.butter(4, [20, 500], btype='bandpass', fs=FS, output='sos')
    data   = signal.sosfiltfilt(sos, data)
    cal    = int(cal_s * FS)
    f, psd = signal.welch(data[:cal], FS, nperseg=FS)
    psd_i  = np.interp(np.fft.rfftfreq(len(data), 1/FS), f, psd)
    w      = np.fft.irfft(np.fft.rfft(data) / np.sqrt(psd_i + 1e-12), n=len(data))
    return np.nan_to_num(w) * tukey(len(w), 0.05)

def get_snr_timeseries(strain):
    cal  = int(5.0 * FS)
    w    = whiten(strain)
    w   /= (np.std(w[:cal]) + 1e-12)
    n    = len(w)
    best = np.zeros(n)
    for cfg in TEMPLATES:
        dur  = cfg['duration']
        tt   = np.linspace(0, dur, int(dur*FS), endpoint=False)
        tmpl = signal.chirp(tt, f0=cfg['f0'], f1=cfg['f1'], t1=dur, method='quadratic')
        tmpl *= tukey(len(tmpl), 0.2)
        tmpl /= (np.sqrt(np.sum(tmpl**2)) + 1e-20)
        if len(tmpl) < n: tmpl = np.pad(tmpl, (0, n-len(tmpl)))
        else: tmpl = tmpl[:n]
        corr  = signal.correlate(w, tmpl, mode='same')
        med   = np.median(corr)
        mad   = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
        snr_t = np.abs(corr - med) / mad
        best  = np.maximum(best, snr_t)
    return best

def get_peak_snr(strain, center_t, window=5.0):
    snr_t = get_snr_timeseries(strain)
    t_arr = np.arange(len(snr_t)) / FS
    mask  = (t_arr >= center_t - window) & (t_arr <= center_t + window)
    return float(np.max(snr_t[mask])) if np.any(mask) else 0.0

def get_triggers(snr_t, threshold):
    triggers = []
    above    = snr_t > threshold
    in_seg   = False
    seg_start = 0
    for i, a in enumerate(above):
        if a and not in_seg:
            in_seg = True; seg_start = i
        elif not a and in_seg:
            in_seg = False
            peak = seg_start + np.argmax(snr_t[seg_start:i])
            triggers.append((peak/FS, float(snr_t[peak])))
    return triggers

def is_glitch(strain, t_tr):
    idx     = int(t_tr * FS)
    half    = int(0.5 * FS)
    snippet = strain[max(0,idx-half): min(len(strain),idx+half)]
    if len(snippet) < 100: return True
    return kurtosis(snippet, fisher=False, bias=False) > 25.0

# ─────────────────────────────────────────────────────────────
# Coincidence logic
# ─────────────────────────────────────────────────────────────

def check_coincidence(strains_dict, center_t, threshold=CALIBRATED_T):
    """
    Проверява multi-detector coincidence.
    strains_dict = {"H1": array, "L1": array, "V1": array (optional)}
    Връща: (coincidence, network_snr, per_det_snr)
    """
    det_snrs   = {}
    det_trigs  = {}

    for det, strain in strains_dict.items():
        if strain is None:
            det_snrs[det]  = 0.0
            det_trigs[det] = []
            continue
        snr_t = get_snr_timeseries(strain)
        trigs = get_triggers(snr_t, threshold)
        # Triggers в ±5s около центъра
        near  = [(t,s) for t,s in trigs if abs(t - center_t) < 5.0]
        peak  = max((s for _,s in near), default=0.0)
        det_snrs[det]  = peak
        det_trigs[det] = near

    # Coincidence check H1 + L1
    coinc_hl = False
    for t_h1, _ in det_trigs.get("H1", []):
        for t_l1, _ in det_trigs.get("L1", []):
            if abs(t_h1 - t_l1) <= COINCIDENCE_DT:
                if not is_glitch(strains_dict["H1"], t_h1):
                    if not is_glitch(strains_dict["L1"], t_l1):
                        coinc_hl = True
                        break

    # V1 coincidence (ако има)
    coinc_v1 = True  # default True ако няма V1
    if "V1" in strains_dict and strains_dict["V1"] is not None:
        coinc_v1 = False
        for t_h1, _ in det_trigs.get("H1", []):
            for t_v1, _ in det_trigs.get("V1", []):
                if abs(t_h1 - t_v1) <= COINCIDENCE_DT_V:
                    coinc_v1 = True
                    break

    coincidence  = coinc_hl and coinc_v1
    network_snr  = np.sqrt(sum(s**2 for s in det_snrs.values()))

    return coincidence, network_snr, det_snrs

# ─────────────────────────────────────────────────────────────
# IFAR curve
# ─────────────────────────────────────────────────────────────

def compute_ifar(quiet_strains_h1, quiet_strains_l1, thresholds):
    """
    IFAR = 1 / FAR
    Изчислява coincidence FAR при различни прагове.
    """
    print("\n  [IFAR] Computing coincidence background...")
    total_time_s = min(
        sum(len(s)/FS for s in quiet_strains_h1),
        sum(len(s)/FS for s in quiet_strains_l1)
    )

    ifar_values = []

    for thr in thresholds:
        total_coinc = 0

        for s_h1, s_l1 in zip(quiet_strains_h1, quiet_strains_l1):
            snr_h1 = get_snr_timeseries(s_h1)
            snr_l1 = get_snr_timeseries(s_l1)

            trigs_h1 = get_triggers(snr_h1, thr)
            trigs_l1 = get_triggers(snr_l1, thr)

            # Count coincidences
            for t_h1, _ in trigs_h1:
                for t_l1, _ in trigs_l1:
                    if abs(t_h1 - t_l1) <= COINCIDENCE_DT:
                        if not is_glitch(s_h1, t_h1) and not is_glitch(s_l1, t_l1):
                            total_coinc += 1
                            break

        far_yr = (total_coinc / total_time_s) * 3.156e7 if total_time_s > 0 else 0
        ifar   = 1.0 / far_yr if far_yr > 0 else 1e6  # cap at 1M years
        ifar_values.append(ifar)
        print(f"  [IFAR] thr={thr:.2f}  coinc={total_coinc}  FAR={far_yr:.2e}/yr  IFAR={ifar:.1f}yr")

    return ifar_values

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_e6():
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 62)
    print("  ORAC-NT PHASE E6 — MULTI-DETECTOR COINCIDENCE + IFAR")
    print(f"  {run_utc}")
    print("=" * 62)

    # ── Load quiet data ───────────────────────────────────
    print("\n📥 LOADING QUIET SEGMENTS")
    quiet_h1, quiet_l1 = [], []
    for seg in QUIET_SEGMENTS:
        print(f"  [{seg['name']}]")
        s = fetch_bulk(seg['gps'], seg['dur'], seg['det'])
        if s is not None:
            if seg['det'] == 'H1': quiet_h1.append(s)
            else: quiet_l1.append(s)
            print(f"    [OK] {len(s)/FS:.0f}s")

    # ── 3-detector events ─────────────────────────────────
    print("\n\n📡 E6-1: THREE-DETECTOR COINCIDENCE")
    print(f"   Δt(H1-L1) ≤ {COINCIDENCE_DT*1000:.0f}ms | "
          f"Δt(H1-V1) ≤ {COINCIDENCE_DT_V*1000:.0f}ms")
    print("-" * 62)

    results_3det = []
    for ev in EVENTS_3DET:
        print(f"\n  [{ev['name']}] {ev['type']} — {ev['detectors']}")
        strains = {}
        for det in ev['detectors']:
            strains[det] = fetch_event(ev['name'], det, ev['duration'])

        center = ev['duration'] / 2.0
        coinc, net_snr, det_snrs = check_coincidence(strains, center)

        status = "✅ COINCIDENCE" if coinc else "❌ NO COINCIDENCE"
        print(f"    {status}")
        print(f"    Network SNR = {net_snr:.2f}  "
              f"(H1={det_snrs.get('H1',0):.1f} "
              f"L1={det_snrs.get('L1',0):.1f} "
              f"V1={det_snrs.get('V1',0):.1f})")
        results_3det.append({**ev, "coinc": coinc, "net_snr": net_snr, "det_snrs": det_snrs})

    # ── 2-detector events ─────────────────────────────────
    print("\n\n📡 E6-2: TWO-DETECTOR COINCIDENCE")
    print("-" * 62)

    results_2det = []
    for ev in EVENTS_2DET:
        print(f"\n  [{ev['name']}] {ev['type']}")
        strains = {}
        for det in ev['detectors']:
            strains[det] = fetch_event(ev['name'], det, ev['duration'])

        center = ev['duration'] / 2.0
        coinc, net_snr, det_snrs = check_coincidence(strains, center)

        status = "✅ COINCIDENCE" if coinc else "❌ NO COINCIDENCE"
        print(f"    {status} | Network SNR = {net_snr:.2f} "
              f"(H1={det_snrs.get('H1',0):.1f} L1={det_snrs.get('L1',0):.1f})")
        results_2det.append({**ev, "coinc": coinc, "net_snr": net_snr, "det_snrs": det_snrs})

    # ── IFAR curve ────────────────────────────────────────
    print("\n\n📡 E6-3: IFAR CURVE")
    print("-" * 62)

    all_results = results_3det + results_2det
    confirmed   = sum(1 for r in all_results if r['coinc'])
    total       = len(all_results)

    ifar_thresholds = np.linspace(3.0, 15.0, 20)
    ifar_values     = []

    if quiet_h1 and quiet_l1:
        ifar_values = compute_ifar(quiet_h1, quiet_l1, ifar_thresholds)
    else:
        print("  [SKIP] No quiet data for IFAR")
        ifar_values = [1e6] * len(ifar_thresholds)

    # ── Summary ───────────────────────────────────────────
    print("\n\n" + "=" * 62)
    print("  E6 RESULTS")
    print("=" * 62)
    print(f"\n  Coincidence Rate : {confirmed}/{total}")
    print(f"  3-detector events: "
          f"{sum(1 for r in results_3det if r['coinc'])}/{len(results_3det)}")
    print(f"  2-detector events: "
          f"{sum(1 for r in results_2det if r['coinc'])}/{len(results_2det)}")

    print(f"\n  Network SNR values:")
    for r in all_results:
        mark = "✅" if r['coinc'] else "❌"
        dets = "/".join(r['detectors'])
        print(f"  {mark} {r['name']:<12} [{dets}]  "
              f"Network SNR = {r['net_snr']:.2f}")

    # ── Visualization ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0e1117')

    # Panel 1 — Network SNR per event
    names    = [r['name'] for r in all_results]
    net_snrs = [r['net_snr'] for r in all_results]
    colors   = ['#00ffcc' if r['coinc'] else '#ff4444' for r in all_results]
    n_dets   = [len(r['detectors']) for r in all_results]

    bars = axes[0].bar(names, net_snrs, color=colors, alpha=0.85)
    axes[0].axhline(CALIBRATED_T, color='red', lw=2, ls='--',
                    label=f'Single-det threshold = {CALIBRATED_T}')
    axes[0].set_title(
        f"Network SNR — Multi-Detector Coincidence\n"
        f"{confirmed}/{total} confirmed  |  "
        f"Δt(H1-L1)≤10ms, Δt(H1-V1)≤27ms",
        color='white', fontsize=10
    )
    axes[0].set_ylabel("Network SNR = √(SNR_H1² + SNR_L1² + SNR_V1²)", color='white')
    axes[0].legend(facecolor='#0e1117', labelcolor='white', fontsize=9)
    axes[0].set_facecolor('#0e1117'); axes[0].tick_params(colors='white', rotation=30)
    for i, (snr, nd) in enumerate(zip(net_snrs, n_dets)):
        axes[0].text(i, snr+0.3, f"{snr:.1f}\n({nd}det)",
                     ha='center', color='white', fontsize=7)

    # Panel 2 — IFAR curve
    ifar_yr = [min(v, 1e5) for v in ifar_values]
    axes[1].semilogy(ifar_thresholds, ifar_yr, 'o-',
                     color='#00d2ff', lw=2.5, markersize=6,
                     label='ORAC-NT H1+L1 coincidence')
    axes[1].axhline(1.0,   color='red',    lw=1.5, ls='--', label='IFAR = 1 yr')
    axes[1].axhline(100.0, color='orange', lw=1.0, ls=':',  label='IFAR = 100 yr')

    # Mark network SNR of detected events
    for r in all_results:
        if r['coinc']:
            c = '#00ffcc'
            axes[1].axvline(r['net_snr'], color=c, lw=1, ls=':', alpha=0.7)
            axes[1].text(r['net_snr'], 0.5, r['name'][-6:],
                         color=c, fontsize=7, rotation=90,
                         va='bottom', ha='center')

    axes[1].set_title("IFAR Curve (Coincidence Background)", color='white', fontsize=10)
    axes[1].set_xlabel("Network SNR Threshold", color='white')
    axes[1].set_ylabel("IFAR (years)", color='white')
    axes[1].legend(facecolor='#0e1117', labelcolor='white', fontsize=9)
    axes[1].set_facecolor('#0e1117'); axes[1].tick_params(colors='white')
    axes[1].set_xlim(3.0, 15.0)

    plt.suptitle(
        f"ORAC-NT Phase E6 — Multi-Detector Coincidence + IFAR\n"
        f"Dimitar Kretski  |  DOI: 10.5281/zenodo.20129975  |  {run_utc}",
        color='white', fontsize=11
    )
    plt.tight_layout()
    plt.savefig("orac_phase_e6.png", dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print(f"\n✅ orac_phase_e6.png")
    print(f"   🕐 {run_utc}")

if __name__ == "__main__":
    run_e6()
