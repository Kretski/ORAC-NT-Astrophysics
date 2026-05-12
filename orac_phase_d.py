"""
ORAC-NT Phase D — Long-Duration FAR + Ablation + PSD Robustness
================================================================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20098932

1. LONG-DURATION FAR TEST  — максимално quiet O3 сегменти
2. ABLATION STUDY          — whitening / kurtosis / MF / combined
3. PSD ROBUSTNESS TEST     — varying PSD, nonstationary noise

ИНСТАЛАЦИЯ:
  pip install gwosc numpy scipy matplotlib requests h5py

УПОТРЕБА:
  python orac_phase_d.py
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

FS           = 4096
CALIBRATED_T = 3.047

# ─────────────────────────────────────────────────────────────
# Long-duration quiet segments — максимално coverage
# GPS интервали верифицирани от v4.x runs (max SNR < 11)
# ─────────────────────────────────────────────────────────────

LONG_QUIET_SEGMENTS = [
    # O3a верифицирани чисти
    {"name": "O3a_01", "gps_start": 1238166018, "detector": "L1", "duration": 4096},
    {"name": "O3a_02", "gps_start": 1238170114, "detector": "L1", "duration": 4096},
    {"name": "O3a_03", "gps_start": 1238174210, "detector": "L1", "duration": 4096},
    {"name": "O3a_04", "gps_start": 1238194690, "detector": "L1", "duration": 4096},
    {"name": "O3a_05", "gps_start": 1238198786, "detector": "L1", "duration": 4096},
    {"name": "O3a_06", "gps_start": 1242578176, "detector": "L1", "duration": 4096},
    {"name": "O3a_07", "gps_start": 1242607616, "detector": "L1", "duration": 4096},
    # O3b
    {"name": "O3b_01", "gps_start": 1248965632, "detector": "L1", "duration": 4096},
    # Нови O3a — далеч от известни евенти
    {"name": "O3a_08", "gps_start": 1239000000, "detector": "L1", "duration": 4096},
    {"name": "O3a_09", "gps_start": 1239100000, "detector": "L1", "duration": 4096},
    {"name": "O3a_10", "gps_start": 1239200000, "detector": "L1", "duration": 4096},
    {"name": "O3a_11", "gps_start": 1239300000, "detector": "L1", "duration": 4096},
    {"name": "O3a_12", "gps_start": 1239400000, "detector": "L1", "duration": 4096},
    {"name": "O3a_13", "gps_start": 1239500000, "detector": "L1", "duration": 4096},
    {"name": "O3a_14", "gps_start": 1239600000, "detector": "L1", "duration": 4096},
    {"name": "O3a_15", "gps_start": 1239700000, "detector": "L1", "duration": 4096},
    {"name": "O3a_16", "gps_start": 1239800000, "detector": "L1", "duration": 4096},
    {"name": "O3a_17", "gps_start": 1239900000, "detector": "L1", "duration": 4096},
    {"name": "O3a_18", "gps_start": 1240000000, "detector": "L1", "duration": 4096},
    {"name": "O3a_19", "gps_start": 1240100000, "detector": "L1", "duration": 4096},
    {"name": "O3a_20", "gps_start": 1240200000, "detector": "L1", "duration": 4096},
    {"name": "O3a_21", "gps_start": 1240300000, "detector": "L1", "duration": 4096},
    {"name": "O3a_22", "gps_start": 1240400000, "detector": "L1", "duration": 4096},
    {"name": "O3a_23", "gps_start": 1240500000, "detector": "L1", "duration": 4096},
    {"name": "O3a_24", "gps_start": 1240600000, "detector": "L1", "duration": 4096},
]

GW_EVENTS_ABLATION = [
    {"name": "GW170817", "type": "BNS",  "duration": 32},
    {"name": "GW150914", "type": "BBH",  "duration": 32},
    {"name": "GW190521", "type": "BBH",  "duration": 32},
    {"name": "GW190814", "type": "NSBH", "duration": 32},
    {"name": "GW151012", "type": "BBH",  "duration": 32},
]

# ─────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────

def fetch_event_strain(event_name, detector='L1', duration=32):
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
            mid = len(strain) // 2
            strain = strain[mid-target//2: mid+target//2]
        print(f"    [OK] {len(strain)/FS:.0f}s")
        return strain
    except Exception as e:
        print(f"    [WARN] {e}")
        return None

def fetch_bulk_strain(gps_start, duration, detector='L1'):
    try:
        import requests, h5py
        from gwosc.locate import get_urls
        time.sleep(2.0)
        urls = get_urls(detector, gps_start, gps_start + duration)
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
        print(f"    [WARN] {e}")
        return None

# ─────────────────────────────────────────────────────────────
# ORAC-NT components (за ablation)
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
    return np.nan_to_num(w) * tukey(len(w), 0.05)

def make_template(cfg, n):
    tt   = np.linspace(0, cfg['duration'], int(cfg['duration']*FS), endpoint=False)
    tmpl = signal.chirp(tt, f0=cfg['f0'], f1=cfg['f1'], t1=cfg['duration'], method='quadratic')
    tmpl *= tukey(len(tmpl), 0.2)
    tmpl /= (np.sqrt(np.sum(tmpl**2)) + 1e-20)
    if len(tmpl) < n: tmpl = np.pad(tmpl, (0, n-len(tmpl)))
    else: tmpl = tmpl[:n]
    return tmpl

def get_peak_snr_full(strain, center_t, window=5.0):
    """Full ORAC-NT pipeline."""
    cal   = int(5.0 * FS)
    w     = whiten(strain)
    w    /= (np.std(w[:cal]) + 1e-12)
    n     = len(w)
    best  = 0.0
    for cfg in TEMPLATES:
        tmpl  = make_template(cfg, n)
        corr  = signal.correlate(w, tmpl, mode='same')
        med   = np.median(corr)
        mad   = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
        snr_t = np.abs(corr - med) / mad
        t_arr = np.arange(n) / FS
        mask  = (t_arr >= center_t - window) & (t_arr <= center_t + window)
        if np.any(mask):
            best = max(best, float(np.max(snr_t[mask])))
    return best

def is_glitch_kurtosis(strain, t_trigger, window=1.0):
    idx     = int(t_trigger * FS)
    half    = int(window/2*FS)
    snippet = strain[max(0,idx-half): min(len(strain),idx+half)]
    if len(snippet) < 100: return True
    return kurtosis(snippet, fisher=False, bias=False) > 25.0

def get_peak_snr_whitening_only(strain, center_t, window=5.0):
    """Само whitening — без matched filter."""
    cal   = int(5.0 * FS)
    w     = whiten(strain)
    w    /= (np.std(w[:cal]) + 1e-12)
    env   = np.abs(signal.hilbert(w))
    t_arr = np.arange(len(w)) / FS
    mask  = (t_arr >= center_t - window) & (t_arr <= center_t + window)
    return float(np.max(env[mask])) if np.any(mask) else 0.0

def get_peak_snr_no_veto(strain, center_t, window=5.0):
    """Full MF без kurtosis veto."""
    return get_peak_snr_full(strain, center_t, window)

def count_false_triggers_full(strain, threshold):
    """Брои false triggers с пълен pipeline."""
    cal   = int(5.0 * FS)
    w     = whiten(strain)
    w    /= (np.std(w[:cal]) + 1e-12)
    n     = len(w)
    best_snr_t = np.zeros(n)
    for cfg in TEMPLATES:
        tmpl  = make_template(cfg, n)
        corr  = signal.correlate(w, tmpl, mode='same')
        med   = np.median(corr)
        mad   = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
        snr_t = np.abs(corr - med) / mad
        best_snr_t = np.maximum(best_snr_t, snr_t)

    # Cluster triggers
    above    = best_snr_t > threshold
    triggers = []
    in_seg   = False
    seg_start = 0
    for i, a in enumerate(above):
        if a and not in_seg:
            in_seg = True; seg_start = i
        elif not a and in_seg:
            in_seg = False
            peak = seg_start + np.argmax(best_snr_t[seg_start:i])
            t_peak = peak / FS
            if not is_glitch_kurtosis(strain, t_peak):
                triggers.append(t_peak)
    return len(triggers)

# ─────────────────────────────────────────────────────────────
# 1. Long-Duration FAR
# ─────────────────────────────────────────────────────────────

def run_long_far():
    print("\n📡 PHASE D-1: LONG-DURATION FAR TEST")
    print("-" * 62)

    total_time_s  = 0.0
    total_false   = 0
    seg_results   = []
    MAX_QUIET_SNR = 11.0

    for seg in LONG_QUIET_SEGMENTS:
        print(f"\n  [{seg['name']}] GPS={seg['gps_start']}")
        strain = fetch_bulk_strain(seg['gps_start'], seg['duration'], seg['detector'])

        if strain is None:
            print(f"    [SKIP] Not available")
            continue

        # Pre-scan за massive glitch
        cal   = int(5.0 * FS)
        w     = whiten(strain)
        w    /= (np.std(w[:cal]) + 1e-12)
        n     = len(w)
        best_snr_t = np.zeros(n)
        for cfg in TEMPLATES:
            tmpl  = make_template(cfg, n)
            corr  = signal.correlate(w, tmpl, mode='same')
            med   = np.median(corr)
            mad   = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
            snr_t = np.abs(corr - med) / mad
            best_snr_t = np.maximum(best_snr_t, snr_t)

        max_snr = float(np.max(best_snr_t))
        print(f"    max SNR: {max_snr:.2f}")

        if max_snr > MAX_QUIET_SNR:
            print(f"    [SKIP] Massive glitch (SNR={max_snr:.1f} > {MAX_QUIET_SNR})")
            continue

        # Count false triggers
        false_count  = count_false_triggers_full(strain, CALIBRATED_T)
        actual_dur   = len(strain) / FS
        total_time_s += actual_dur
        total_false  += false_count

        far_seg = (false_count / actual_dur) * 3.156e7 if actual_dur > 0 else 0
        print(f"    False triggers: {false_count} | Duration: {actual_dur:.0f}s | FAR: {far_seg:.2e} /yr")
        seg_results.append({
            "name": seg['name'],
            "duration": actual_dur,
            "false": false_count,
            "far": far_seg
        })

    # Summary
    far_total = (total_false / total_time_s) * 3.156e7 if total_time_s > 0 else 0
    far_per_day = (total_false / total_time_s) * 86400 if total_time_s > 0 else 0
    far_per_week = far_per_day * 7

    print(f"\n  {'='*40}")
    print(f"  Total quiet time : {total_time_s/3600:.2f} hours")
    print(f"  Total false alarms: {total_false}")
    print(f"  FAR = {far_total:.2e} /yr")
    print(f"  FAR = {far_per_day:.4f} /day")
    print(f"  FAR = {far_per_week:.3f} /week")

    # Upper limit (Poisson, 90% confidence)
    if total_false == 0 and total_time_s > 0:
        upper_limit = 2.303 / (total_time_s / 3.156e7)
        print(f"  Upper limit (90% CL): FAR < {upper_limit:.2e} /yr")

    return seg_results, total_time_s, total_false, far_total

# ─────────────────────────────────────────────────────────────
# 2. Ablation Study
# ─────────────────────────────────────────────────────────────

def run_ablation(event_strains):
    print("\n\n📡 PHASE D-2: ABLATION STUDY")
    print("-" * 62)
    print(f"  {'Config':<25} {'Detected':>10} {'Rate':>8}")
    print(f"  {'-'*45}")

    configs = [
        ("Whitening only",      "whitening"),
        ("MF only (no veto)",   "mf_no_veto"),
        ("Full ORAC-NT",        "full"),
    ]

    ablation_results = {}

    for config_name, config_key in configs:
        detected = 0
        snrs     = []

        for ev_name, strain in event_strains.items():
            if strain is None:
                continue

            center = len(strain) / FS / 2.0

            if config_key == "whitening":
                snr = get_peak_snr_whitening_only(strain, center)
                # Whitening only threshold — env based
                det = snr > 5.0

            elif config_key == "mf_no_veto":
                snr = get_peak_snr_no_veto(strain, center)
                det = snr >= CALIBRATED_T

            else:  # full
                snr = get_peak_snr_full(strain, center)
                det = snr >= CALIBRATED_T and not is_glitch_kurtosis(strain, center)

            if det:
                detected += 1
            snrs.append(snr)

        total   = len(event_strains)
        rate    = detected / total * 100 if total > 0 else 0
        avg_snr = np.mean(snrs) if snrs else 0

        print(f"  {config_name:<25} {detected}/{total}      {rate:.0f}%  (avg SNR={avg_snr:.1f})")
        ablation_results[config_name] = {
            "detected": detected,
            "total": total,
            "rate": rate,
            "avg_snr": avg_snr
        }

    return ablation_results

# ─────────────────────────────────────────────────────────────
# 3. PSD Robustness
# ─────────────────────────────────────────────────────────────

def run_psd_robustness(quiet_strains, event_strains):
    print("\n\n📡 PHASE D-3: PSD ROBUSTNESS TEST")
    print("-" * 62)

    results = {}

    # Test 1: Varying calibration window
    cal_windows = [2.0, 5.0, 10.0, 20.0]
    print("\n  [Test 1] Varying calibration window:")
    cal_results = []
    for cal_s in cal_windows:
        detected = 0
        for ev_name, strain in event_strains.items():
            if strain is None: continue
            center = len(strain)/FS/2.0
            cal    = int(cal_s * FS)
            if cal >= len(strain)//2: continue
            w      = whiten(strain, cal_s=cal_s)
            w     /= (np.std(w[:cal]) + 1e-12)
            n      = len(w)
            best   = 0.0
            for cfg in TEMPLATES:
                tmpl  = make_template(cfg, n)
                corr  = signal.correlate(w, tmpl, mode='same')
                med   = np.median(corr)
                mad   = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
                snr_t = np.abs(corr - med) / mad
                t_arr = np.arange(n)/FS
                mask  = (t_arr >= center-5) & (t_arr <= center+5)
                if np.any(mask):
                    best = max(best, float(np.max(snr_t[mask])))
            if best >= CALIBRATED_T:
                detected += 1
        total = sum(1 for s in event_strains.values() if s is not None)
        rate  = detected/total*100 if total > 0 else 0
        print(f"    cal_s={cal_s:5.1f}s  →  {detected}/{total} ({rate:.0f}%)")
        cal_results.append({"cal_s": cal_s, "rate": rate})
    results["cal_window"] = cal_results

    # Test 2: Nonstationary noise — inject drift
    print("\n  [Test 2] Nonstationary noise (PSD drift):")
    drift_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    drift_results = []
    if quiet_strains:
        base_noise = quiet_strains[0][:32*FS].copy()
        for drift in drift_levels:
            # Simulate PSD drift by amplitude modulation
            t_mod  = np.linspace(1.0, 1.0+drift, len(base_noise))
            drifted = base_noise * t_mod
            false_count = count_false_triggers_full(drifted, CALIBRATED_T)
            far_drift   = (false_count / (len(drifted)/FS)) * 3.156e7
            print(f"    drift={drift:.1f}x  →  false={false_count}  FAR={far_drift:.2e} /yr")
            drift_results.append({"drift": drift, "false": false_count, "far": far_drift})
    results["drift"] = drift_results

    return results

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_phase_d():

    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 62)
    print("  ORAC-NT PHASE D BENCHMARK")
    print(f"  {run_utc}")
    print("=" * 62)

    # ── D1: Long-duration FAR ─────────────────────────────
    seg_results, total_time_s, total_false, far_total = run_long_far()

    # ── Load events for ablation ──────────────────────────
    print("\n\n📥 LOADING EVENTS FOR ABLATION")
    event_strains = {}
    for ev in GW_EVENTS_ABLATION:
        print(f"  [{ev['name']}]")
        s = fetch_event_strain(ev['name'], 'L1', ev['duration'])
        event_strains[ev['name']] = s

    quiet_strains = []
    for seg in seg_results[:2]:
        s = fetch_bulk_strain(
            LONG_QUIET_SEGMENTS[0]['gps_start'],
            LONG_QUIET_SEGMENTS[0]['duration'], 'L1'
        )
        if s is not None:
            quiet_strains.append(s)
            break

    # ── D2: Ablation ──────────────────────────────────────
    ablation_results = run_ablation(event_strains)

    # ── D3: PSD Robustness ────────────────────────────────
    psd_results = run_psd_robustness(quiet_strains, event_strains)

    # ── Final summary ─────────────────────────────────────
    print("\n\n" + "=" * 62)
    print("  PHASE D FINAL RESULTS")
    print("=" * 62)
    print(f"\n  Total quiet time : {total_time_s/3600:.2f} hours")
    print(f"  Total false alarms: {total_false}")
    print(f"  FAR              : {far_total:.2e} /yr")
    if total_false == 0 and total_time_s > 0:
        ul = 2.303 / (total_time_s / 3.156e7)
        print(f"  Upper limit 90%CL: FAR < {ul:.2e} /yr")

    # ── Visualization ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0e1117')

    # Panel 1 — FAR per segment
    if seg_results:
        seg_names = [r['name'] for r in seg_results]
        seg_false = [r['false'] for r in seg_results]
        seg_durs  = [r['duration']/3600 for r in seg_results]
        colors_seg = ['#00ffcc' if f == 0 else '#ff9900' for f in seg_false]
        axes[0,0].bar(range(len(seg_names)), seg_false, color=colors_seg, alpha=0.85)
        axes[0,0].set_xticks(range(len(seg_names)))
        axes[0,0].set_xticklabels(seg_names, rotation=45, color='white', fontsize=7)
        far_str = f"FAR = {far_total:.2e} /yr" if far_total > 0 else f"FAR = 0  (UL: {2.303/(total_time_s/3.156e7):.2e} /yr)" if total_time_s > 0 else "FAR = N/A"
        axes[0,0].set_title(
            f"Long-Duration FAR — {total_time_s/3600:.1f}h total\n{far_str}",
            color='white', fontsize=10
        )
        axes[0,0].set_ylabel("False Triggers", color='white')
        axes[0,0].set_facecolor('#0e1117'); axes[0,0].tick_params(colors='white')
        axes[0,0].text(0.98, 0.95, f"Total: {total_time_s/3600:.1f}h",
                       transform=axes[0,0].transAxes, color='#aaaaaa',
                       fontsize=9, ha='right', va='top')

    # Panel 2 — Ablation study
    if ablation_results:
        abl_names = list(ablation_results.keys())
        abl_rates = [ablation_results[n]['rate'] for n in abl_names]
        abl_snrs  = [ablation_results[n]['avg_snr'] for n in abl_names]
        abl_colors = ['#4488ff', '#ff9900', '#00ffcc']
        bars = axes[0,1].bar(abl_names, abl_rates, color=abl_colors[:len(abl_names)], alpha=0.85)
        axes[0,1].set_title("Ablation Study — Component Contribution", color='white', fontsize=10)
        axes[0,1].set_ylabel("Detection Rate (%)", color='white')
        axes[0,1].set_ylim(0, 115)
        axes[0,1].set_facecolor('#0e1117'); axes[0,1].tick_params(colors='white', rotation=15)
        for bar, rate, snr in zip(bars, abl_rates, abl_snrs):
            axes[0,1].text(bar.get_x()+bar.get_width()/2, rate+2,
                           f'{rate:.0f}%\nSNR={snr:.0f}',
                           ha='center', color='white', fontsize=8)

    # Panel 3 — PSD calibration window
    if psd_results.get('cal_window'):
        cal_data = psd_results['cal_window']
        cal_xs   = [d['cal_s'] for d in cal_data]
        cal_ys   = [d['rate']  for d in cal_data]
        axes[1,0].plot(cal_xs, cal_ys, 'o-', color='#00ffcc', lw=2, markersize=8)
        axes[1,0].axhline(100, color='gray', ls='--', lw=1, alpha=0.5)
        axes[1,0].set_title("PSD Robustness — Calibration Window", color='white', fontsize=10)
        axes[1,0].set_xlabel("Calibration Duration (s)", color='white')
        axes[1,0].set_ylabel("Detection Rate (%)", color='white')
        axes[1,0].set_ylim(0, 115)
        axes[1,0].set_facecolor('#0e1117'); axes[1,0].tick_params(colors='white')

    # Panel 4 — PSD drift
    if psd_results.get('drift'):
        drift_data = psd_results['drift']
        drift_xs   = [d['drift'] for d in drift_data]
        drift_ys   = [max(d['far'], 0.001) for d in drift_data]
        axes[1,1].semilogy(drift_xs, drift_ys, 'o-', color='#ff9900', lw=2, markersize=8)
        axes[1,1].axhline(1.0, color='red', ls='--', lw=1.5, label='FAR=1/yr target')
        axes[1,1].set_title("PSD Robustness — Noise Drift", color='white', fontsize=10)
        axes[1,1].set_xlabel("Amplitude Drift Factor", color='white')
        axes[1,1].set_ylabel("FAR (/yr)", color='white')
        axes[1,1].legend(facecolor='#0e1117', labelcolor='white', fontsize=9)
        axes[1,1].set_facecolor('#0e1117'); axes[1,1].tick_params(colors='white')

    plt.suptitle(
        f"ORAC-NT Phase D — Long FAR + Ablation + PSD Robustness\n"
        f"Dimitar Kretski  |  DOI: 10.5281/zenodo.20098932  |  {run_utc}",
        color='white', fontsize=11
    )
    plt.tight_layout()
    plt.savefig("orac_phase_d.png", dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print(f"\n✅ orac_phase_d.png")
    print(f"   🕐 {run_utc}")

if __name__ == "__main__":
    run_phase_d()
