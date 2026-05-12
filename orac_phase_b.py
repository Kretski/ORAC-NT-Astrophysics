"""
ORAC-NT Phase B Benchmark v2
==========================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20098932

1. INJECTION CAMPAIGN  — BNS/BBH/NSBH в реален O3 шум
2. ROC CURVES          — TPR vs FPR при различни SNR прагове
3. SENSITIVITY SWEEP   — Detection efficiency vs injection SNR (4..10)
4. EVENT GENERALIZATION — 4 нови GW евента

ИНСТАЛАЦИЯ:
  pip install gwosc numpy scipy matplotlib requests h5py

УПОТРЕБА:
  python orac_phase_b.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
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

FS           = 4096
CALIBRATED_T = 3.047   # от v4.2 benchmark

# ─────────────────────────────────────────────────────────────
# Event generalization — нови евенти
# ─────────────────────────────────────────────────────────────

GW_EVENTS_EXTENDED = [
    # От v4.2 (верифицирани)
    {"name": "GW170817", "type": "BNS",  "detector": "L1", "duration": 32},
    {"name": "GW150914", "type": "BBH",  "detector": "L1", "duration": 32},
    {"name": "GW151226", "type": "BBH",  "detector": "L1", "duration": 32},
    {"name": "GW190521", "type": "BBH",  "detector": "L1", "duration": 32},
    {"name": "GW190814", "type": "NSBH", "detector": "L1", "duration": 32},
    # Нови за generalization
    {"name": "GW151012", "type": "BBH",  "detector": "L1", "duration": 32},
    {"name": "GW170104", "type": "BBH",  "detector": "L1", "duration": 32},
    {"name": "GW170608", "type": "BBH",  "detector": "L1", "duration": 32},
    {"name": "GW200105", "type": "NSBH", "detector": "L1", "duration": 32},
    {"name": "GW200115", "type": "NSBH", "detector": "L1", "duration": 32},
]

# Тихи сегменти за injection (верифицирани чисти)
QUIET_FOR_INJECTION = [
    {"name": "O3a_q01", "gps_start": 1238166018, "detector": "L1", "duration": 4096},
    {"name": "O3a_q12", "gps_start": 1242578176, "detector": "L1", "duration": 4096},
]

# ─────────────────────────────────────────────────────────────
# Fetch helpers
# ─────────────────────────────────────────────────────────────

def fetch_event_strain(event_name, detector='L1', duration=32):
    try:
        import requests, h5py
        from gwosc.locate import get_event_urls
        urls = get_event_urls(event_name, detector=detector, duration=duration)
        if not urls: return None
        r = requests.get(urls[0], timeout=120, stream=True)
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
            mid = len(strain) // 2
            strain = strain[mid-target//2: mid+target//2]
        print(f"    [OK] {len(strain)/FS:.0f}s @ {FS} Hz")
        return strain
    except Exception as e:
        print(f"    [WARN] {e}")
        return None

def fetch_bulk_strain(gps_start, duration, detector='L1'):
    try:
        import requests, h5py
        from gwosc.locate import get_urls
        urls = get_urls(detector, gps_start, gps_start + duration)
        if not urls: return None
        hdf_url = next((u for u in urls if '4096' in u and 'hdf5' in u), urls[0])
        r = requests.get(hdf_url, timeout=180, stream=True)
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
        print(f"    [OK] {len(strain)/FS:.0f}s @ {FS} Hz")
        return strain
    except Exception as e:
        print(f"    [WARN] {e}")
        return None

# ─────────────────────────────────────────────────────────────
# ORAC-NT core
# ─────────────────────────────────────────────────────────────

TEMPLATES = [
    {"name": "BBH_O1",   "f0": 35,  "f1": 150, "duration": 0.2},
    {"name": "BBH_gen",  "f0": 20,  "f1": 350, "duration": 0.4},
    {"name": "BBH_lite", "f0": 30,  "f1": 450, "duration": 0.5},
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
    w      = np.nan_to_num(w)
    return w * tukey(len(w), 0.05)

def make_template(cfg, n):
    f0, f1, dur = cfg['f0'], cfg['f1'], cfg['duration']
    tt   = np.linspace(0, dur, int(dur*FS), endpoint=False)
    tmpl = signal.chirp(tt, f0=f0, f1=f1, t1=dur, method='quadratic')
    tmpl *= tukey(len(tmpl), 0.2)
    tmpl /= (np.sqrt(np.sum(tmpl**2)) + 1e-20)
    # Pad/crop to n
    if len(tmpl) < n:
        tmpl = np.pad(tmpl, (0, n-len(tmpl)))
    else:
        tmpl = tmpl[:n]
    return tmpl

def chi2_veto(corr, snr, nbands=4):
    n      = len(corr)
    edges  = np.linspace(0, n, nbands+1, dtype=int)
    chi2   = 0.0
    exp    = snr / nbands
    for i in range(nbands):
        band  = corr[edges[i]:edges[i+1]]
        obs   = float(np.max(np.abs(band)))
        chi2 += (obs - exp)**2 / (exp + 1e-20)
    return chi2 / nbands

def get_peak_snr(strain, center_t, window=5.0, threshold=CALIBRATED_T):
    cal   = int(5.0 * FS)
    w     = whiten(strain)
    w    /= (np.std(w[:cal]) + 1e-12)
    n     = len(w)
    best_snr  = 0.0
    best_chi2 = 0.0

    for cfg in TEMPLATES:
        tmpl = make_template(cfg, n)
        corr = signal.correlate(w, tmpl, mode='same')
        med  = np.median(corr)
        mad  = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
        snr_t = np.abs(corr - med) / mad

        t_arr = np.arange(n) / FS
        mask  = (t_arr >= center_t - window) & (t_arr <= center_t + window)
        if not np.any(mask): continue

        pk = float(np.max(snr_t[mask]))
        if pk > best_snr:
            best_snr  = pk
            peak_idx  = np.argmax(snr_t[mask])
            peak_corr = corr[mask]
            best_chi2 = chi2_veto(peak_corr, pk)

    return best_snr, best_chi2

def is_glitch(strain, t_trigger, window=1.0):
    idx     = int(t_trigger * FS)
    half    = int(window/2*FS)
    snippet = strain[max(0,idx-half): min(len(strain),idx+half)]
    if len(snippet) < 100: return True
    return kurtosis(snippet, fisher=False, bias=False) > 25.0

# ─────────────────────────────────────────────────────────────
# Injection campaign
# ─────────────────────────────────────────────────────────────

def inject_signal(noise, target_snr, signal_type='BNS', t_inject=None):
    """
    Инжектира симулиран GW сигнал в реален шум при зададен SNR.
    Връща (strain_with_injection, t_inject, actual_snr).
    """
    n = len(noise)
    if t_inject is None:
        t_inject = n/FS/2.0  # инжектирай в центъра

    # Параметри по тип
    if signal_type == 'BNS':
        f0, f1, dur = 30, 400, 1.0
    elif signal_type == 'BBH':
        f0, f1, dur = 20, 350, 0.4
    else:  # NSBH
        f0, f1, dur = 20, 300, 0.6

    tt   = np.linspace(0, dur, int(dur*FS), endpoint=False)
    tmpl = signal.chirp(tt, f0=f0, f1=f1, t1=dur, method='quadratic')
    tmpl *= tukey(len(tmpl), 0.2)

    # Нормализирай до целевия SNR спрямо whitened noise
    # Whiten noise за по-точна SNR оценка
    sos_w     = signal.butter(4, [20, 500], btype='bandpass', fs=FS, output='sos')
    noise_w   = signal.sosfiltfilt(sos_w, noise)
    noise_rms = np.std(noise_w[int(1.0*FS):int(4.0*FS)]) + 1e-20
    tmpl_rms  = np.sqrt(np.mean(tmpl**2)) + 1e-20
    amplitude  = target_snr * noise_rms / tmpl_rms
    tmpl      *= amplitude

    # Инжектирай
    strain = noise.copy()
    idx    = int(t_inject * FS)
    half   = len(tmpl) // 2
    start  = max(0, idx - half)
    end    = min(n, start + len(tmpl))
    l      = end - start
    strain[start:end] += tmpl[:l]

    return strain, t_inject

def run_injection_campaign(quiet_strains):
    """
    Инжектира сигнали при SNR=4..12 и измерва recovery rate.
    """
    print("\n📡 INJECTION CAMPAIGN")
    print("-" * 62)

    signal_types = ['BNS', 'BBH', 'NSBH']
    snr_levels   = [4, 5, 6, 7, 8, 9, 10, 12]
    n_trials     = 20   # инжекции на SNR ниво

    results          = {st: {snr: [] for snr in snr_levels} for st in signal_types}
    raw_measurements = []

    for sig_type in signal_types:
        print(f"\n  [{sig_type}] injection sweep...")
        for target_snr in snr_levels:
            detected = 0
            for trial in range(n_trials):
                # Вземи случаен 32s прозорец от тихите данни
                noise_src = quiet_strains[trial % len(quiet_strains)]
                start_idx = np.random.randint(0, len(noise_src) - 32*FS)
                noise     = noise_src[start_idx: start_idx + 32*FS].copy()

                strain, t_inj = inject_signal(noise, target_snr, sig_type)

                peak_snr, chi2 = get_peak_snr(strain, t_inj, window=3.0)

                glitch = is_glitch(strain, t_inj)
                det    = (peak_snr >= CALIBRATED_T) and not glitch
                if det:
                    detected += 1
                raw_measurements.append((sig_type, target_snr, float(peak_snr)))

            rate = detected / n_trials
            results[sig_type][target_snr] = rate
            print(f"    SNR={target_snr:2d}  recovery={rate*100:.0f}%  ({detected}/{n_trials})")

    return results, snr_levels, raw_measurements

# ─────────────────────────────────────────────────────────────
# ROC curves
# ─────────────────────────────────────────────────────────────

def run_roc(quiet_strains, inj_results_raw):
    """
    ROC curve базирана на injection campaign.
    signal_snrs = SNR-ове от всички инжекции
    noise_snrs  = SNR-ове от тихи прозорци (без инжекция)
    """
    print("\n\n📡 ROC CURVES (injection-based)")
    print("-" * 62)

    signal_snrs = []
    noise_snrs  = []

    # Signal SNRs — от injection campaign (всички типове, всички SNR нива)
    for item in inj_results_raw:
        sig_type, snr_level, snr_measured = item[0], item[1], float(item[2])
        signal_snrs.append(snr_measured)

    # Noise SNRs — от тихи 32s прозорци без инжекция
    for qs in quiet_strains:
        for i in range(0, len(qs) - 32*FS, 32*FS):
            chunk = qs[i: i + 32*FS]
            snr, _ = get_peak_snr(chunk, 16.0, window=5.0)
            noise_snrs.append(snr)

    print(f"  Signal samples: {len(signal_snrs)}")
    print(f"  Noise samples:  {len(noise_snrs)}")

    if not signal_snrs or not noise_snrs:
        return np.array([0,1]), [0,1], [0,1], 0.5

    all_snrs = sorted(set(signal_snrs + noise_snrs), reverse=True)
    tprs, fprs = [0.0], [0.0]

    for thr in all_snrs:
        tpr = sum(1 for s in signal_snrs if s >= thr) / len(signal_snrs)
        fpr = sum(1 for n in noise_snrs  if n >= thr) / len(noise_snrs)
        tprs.append(tpr)
        fprs.append(fpr)

    tprs.append(1.0); fprs.append(1.0)

    # AUC via trapezoid на sorted FPR
    sorted_pairs = sorted(zip(fprs, tprs))
    s_fprs = [p[0] for p in sorted_pairs]
    s_tprs = [p[1] for p in sorted_pairs]
    auc = float(np.trapz(s_tprs, s_fprs))
    print(f"  AUC = {auc:.3f}")

    return all_snrs, tprs, fprs, auc

# ─────────────────────────────────────────────────────────────
# Main benchmark
# ─────────────────────────────────────────────────────────────

def run_phase_b():

    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 62)
    print("  ORAC-NT PHASE B BENCHMARK")
    print(f"  {run_utc}")
    print("=" * 62)

    # ── Load quiet data ───────────────────────────────────
    print("\n📥 LOADING QUIET SEGMENTS")
    quiet_strains = []
    for seg in QUIET_FOR_INJECTION:
        print(f"\n  [{seg['name']}] GPS={seg['gps_start']}")
        s = fetch_bulk_strain(seg['gps_start'], seg['duration'], seg['detector'])
        if s is not None:
            quiet_strains.append(s)

    if not quiet_strains:
        print("[ERR] No quiet data. Check GWOSC access.")
        return

    # ── Event generalization ──────────────────────────────
    print("\n\n📡 EVENT GENERALIZATION (10 events)")
    print("-" * 62)

    gen_results  = []
    signal_strains = []

    for ev in GW_EVENTS_EXTENDED:
        print(f"\n  [{ev['name']}] {ev['type']}")
        strain = fetch_event_strain(ev['name'], ev['detector'], ev['duration'])
        if strain is None:
            gen_results.append({**ev, "detected": False, "snr": 0.0})
            continue

        center   = len(strain)/FS/2.0
        snr, chi2 = get_peak_snr(strain, center)
        detected = (snr >= CALIBRATED_T) and not is_glitch(strain, center)

        status = "✅ DETECTED" if detected else "❌ MISSED"
        print(f"    {status}  SNR={snr:.2f}  chi2={chi2:.1f}")
        gen_results.append({**ev, "detected": detected, "snr": snr, "chi2": chi2})
        if detected:
            signal_strains.append(strain)

    detected_count = sum(1 for r in gen_results if r['detected'])
    print(f"\n  Generalization: {detected_count}/{len(gen_results)} detected")

    # ── Injection campaign ────────────────────────────────
    inj_results, snr_levels, inj_raw = run_injection_campaign(quiet_strains)

    # ── ROC curves ────────────────────────────────────────
    thresholds, tprs, fprs, auc = run_roc(quiet_strains, inj_raw)

    # ── Visualization ─────────────────────────────────────
    print("\n\n[VIZ] Generating Phase B plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0e1117')

    # Panel 1 — Event generalization
    names  = [r['name'] for r in gen_results]
    snrs   = [r['snr']  for r in gen_results]
    colors = ['#00ffcc' if r['detected'] else '#ff4444' for r in gen_results]
    axes[0,0].bar(names, snrs, color=colors, alpha=0.85)
    axes[0,0].axhline(CALIBRATED_T, color='red', lw=2, ls='--',
                      label=f'Threshold = {CALIBRATED_T}')
    axes[0,0].set_title(f"Event Generalization — {detected_count}/{len(gen_results)} detected",
                        color='white', fontsize=10)
    axes[0,0].set_ylabel("Peak SNR", color='white')
    axes[0,0].legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
    axes[0,0].set_facecolor('#0e1117'); axes[0,0].tick_params(colors='white', rotation=45)
    for i, snr in enumerate(snrs):
        axes[0,0].text(i, snr+0.2, f"{snr:.1f}", ha='center', color='white', fontsize=7)

    # Panel 2 — Injection sensitivity sweep
    colors_inj = {'BNS': '#00ffcc', 'BBH': '#4488ff', 'NSBH': '#ff9900'}
    for sig_type in ['BNS', 'BBH', 'NSBH']:
        rates = [inj_results[sig_type][snr]*100 for snr in snr_levels]
        axes[0,1].plot(snr_levels, rates, 'o-', color=colors_inj[sig_type],
                       lw=2, label=sig_type, markersize=5)
    axes[0,1].axhline(50, color='gray', ls=':', lw=1, alpha=0.5)
    axes[0,1].axhline(90, color='gray', ls=':', lw=1, alpha=0.5)
    axes[0,1].set_title("Injection Recovery Rate vs SNR", color='white', fontsize=10)
    axes[0,1].set_xlabel("Injected SNR", color='white')
    axes[0,1].set_ylabel("Recovery Rate (%)", color='white')
    axes[0,1].legend(facecolor='#0e1117', labelcolor='white', fontsize=9)
    axes[0,1].set_facecolor('#0e1117'); axes[0,1].tick_params(colors='white')
    axes[0,1].set_ylim(0, 105)

    # Panel 3 — ROC curve
    axes[1,0].plot(fprs, tprs, color='#00d2ff', lw=2.5, label=f'ORAC-NT (AUC={auc:.3f})')
    axes[1,0].plot([0,1], [0,1], color='gray', ls='--', lw=1, alpha=0.5, label='Random')
    axes[1,0].set_title("ROC Curve (TPR vs FPR)", color='white', fontsize=10)
    axes[1,0].set_xlabel("False Positive Rate", color='white')
    axes[1,0].set_ylabel("True Positive Rate", color='white')
    axes[1,0].legend(facecolor='#0e1117', labelcolor='white', fontsize=9)
    axes[1,0].set_facecolor('#0e1117'); axes[1,0].tick_params(colors='white')
    axes[1,0].set_xlim(-0.02, 1.02)
    axes[1,0].set_ylim(-0.02, 1.02)

    # Panel 4 — Summary table
    axes[1,1].axis('off')
    axes[1,1].set_facecolor('#0e1117')

    summary = [
        ["Metric", "Value"],
        ["Calibrated threshold", f"{CALIBRATED_T}"],
        ["Event generalization", f"{detected_count}/{len(gen_results)}"],
        ["ROC AUC", f"{auc:.3f}"],
        ["BNS recovery @ SNR=8", f"{inj_results['BNS'][8]*100:.0f}%"],
        ["BBH recovery @ SNR=8", f"{inj_results['BBH'][8]*100:.0f}%"],
        ["NSBH recovery @ SNR=8", f"{inj_results['NSBH'][8]*100:.0f}%"],
        ["FAR (v4.2)", "0.00e+00 /yr"],
        ["Quiet data", "7.96h"],
    ]

    table = axes[1,1].table(
        cellText=summary[1:],
        colLabels=summary[0],
        cellLoc='center',
        loc='center',
        bbox=[0.05, 0.05, 0.90, 0.90]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor('#1a1a2e' if row == 0 else '#0e1117')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#333333')

    axes[1,1].set_title("ORAC-NT Phase B Summary", color='white', fontsize=10)

    plt.suptitle(
        f"ORAC-NT Phase B — Injection Campaign + ROC + Event Generalization\n"
        f"Dimitar Kretski  |  DOI: 10.5281/zenodo.20098932  |  {run_utc}",
        color='white', fontsize=11
    )
    plt.tight_layout()
    plt.savefig("orac_phase_b.png", dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print(f"\n✅ orac_phase_b.png")
    print(f"   🕐 {run_utc}")

if __name__ == "__main__":
    run_phase_b()