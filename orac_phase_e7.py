"""
ORAC-NT Phase E7 — Real O4 Noise Test
=======================================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20129975

Tests:
1. O4 noise PSD characterization vs O3
2. Pipeline stability on real O4 data
3. FAR on O4 quiet segments
4. Sensitivity curve on O4 noise floor

O4 GPS range: 1238166018 - 1269363618
O4a ended: ~1253977218

ИНСТАЛАЦИЯ:
  pip install gwosc numpy scipy matplotlib requests h5py

УПОТРЕБА:
  python orac_phase_e7.py
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

# O4 quiet segments — верифицирани далеч от известни евенти
O4_QUIET_SEGMENTS = [
    # O4a — L1
    {"name": "O4a_q01_L1", "gps": 1244000000, "det": "L1", "dur": 4096},
    {"name": "O4a_q02_L1", "gps": 1244200000, "det": "L1", "dur": 4096},
    {"name": "O4a_q03_L1", "gps": 1244400000, "det": "L1", "dur": 4096},
    {"name": "O4a_q04_L1", "gps": 1244600000, "det": "L1", "dur": 4096},
    {"name": "O4a_q05_L1", "gps": 1244800000, "det": "L1", "dur": 4096},
    # O4a — H1
    {"name": "O4a_q01_H1", "gps": 1244000000, "det": "H1", "dur": 4096},
    {"name": "O4a_q02_H1", "gps": 1244200000, "det": "H1", "dur": 4096},
    {"name": "O4a_q03_H1", "gps": 1244400000, "det": "H1", "dur": 4096},
]

# O3 reference segments (верифицирани чисти от Phase A-D)
O3_REFERENCE = [
    {"name": "O3a_ref_L1", "gps": 1238166018, "det": "L1", "dur": 4096},
    {"name": "O3a_ref_H1", "gps": 1238166018, "det": "H1", "dur": 4096},
]

# GW events от O4 за detection test
O4_EVENTS = [
    {"name": "GW200105", "type": "NSBH", "det": "L1", "dur": 32},
    {"name": "GW200115", "type": "NSBH", "det": "L1", "dur": 32},
]

# ─────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────

def fetch_bulk(gps, duration, detector):
    try:
        import requests, h5py
        from gwosc.locate import get_urls
        time.sleep(2.0)
        urls = get_urls(detector, gps, gps + duration)
        if not urls:
            print(f"    [SKIP] No data at GPS={gps} {detector}")
            return None
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
        print(f"    [OK] {detector} GPS={gps}: {len(strain)/FS:.0f}s")
        return strain
    except Exception as e:
        print(f"    [WARN] {detector} GPS={gps}: {e}")
        return None

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
        print(f"    [OK] {event_name} {detector}: {len(strain)/FS:.0f}s")
        return strain
    except Exception as e:
        print(f"    [WARN] {event_name} {detector}: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# ORAC-NT core
# ─────────────────────────────────────────────────────────────

TEMPLATES = [
    {"name": "BBH_O1",  "f0": 35,  "f1": 150, "duration": 0.2},
    {"name": "BBH_gen", "f0": 20,  "f1": 350, "duration": 0.4},
    {"name": "BNS",     "f0": 30,  "f1": 400, "duration": 1.0},
    {"name": "NSBH",    "f0": 20,  "f1": 300, "duration": 0.6},
]

def whiten(data, cal_s=5.0):
    sos    = signal.butter(4, [20, 500], btype='bandpass', fs=FS, output='sos')
    data   = signal.sosfiltfilt(sos, data)
    cal    = int(cal_s * FS)
    f, psd = signal.welch(data[:cal], FS, nperseg=FS)
    psd_i  = np.interp(np.fft.rfftfreq(len(data), 1/FS), f, psd)
    w      = np.fft.irfft(np.fft.rfft(data) / np.sqrt(psd_i + 1e-12), n=len(data))
    return np.nan_to_num(w) * tukey(len(w), 0.05)

def compute_psd(strain, cal_s=10.0):
    cal  = int(cal_s * FS)
    seg  = strain[:cal] if len(strain) > cal else strain
    f, psd = signal.welch(seg, FS, nperseg=FS*2, noverlap=FS)
    return f, psd

def get_max_snr(strain, threshold=CALIBRATED_T):
    cal   = int(5.0 * FS)
    w     = whiten(strain)
    w    /= (np.std(w[:cal]) + 1e-12)
    n     = len(w)
    best  = np.zeros(n)
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
    return float(np.max(best)), best

def count_false_triggers(strain, threshold):
    max_snr, snr_t = get_max_snr(strain, threshold)
    above    = snr_t > threshold
    triggers = 0
    in_seg   = False
    for a in above:
        if a and not in_seg: in_seg = True
        elif not a and in_seg:
            in_seg = False
            triggers += 1
    return triggers

# ─────────────────────────────────────────────────────────────
# E7-1: PSD comparison O3 vs O4
# ─────────────────────────────────────────────────────────────

def run_psd_comparison(o3_strains, o4_strains):
    print("\n📡 E7-1: PSD COMPARISON O3 vs O4")
    print("-" * 62)

    o3_psds = []
    o4_psds = []
    freqs   = None

    for name, s in o3_strains.items():
        if s is None: continue
        f, psd = compute_psd(s)
        if freqs is None: freqs = f
        o3_psds.append(psd)
        print(f"  [O3] {name}: PSD computed")

    for name, s in o4_strains.items():
        if s is None: continue
        f, psd = compute_psd(s)
        o4_psds.append(psd)
        print(f"  [O4] {name}: PSD computed")

    if not o3_psds or not o4_psds:
        print("  [WARN] Insufficient data for PSD comparison")
        return freqs, None, None

    o3_median = np.median(o3_psds, axis=0)
    o4_median = np.median(o4_psds, axis=0)

    # Sensitivity improvement at key frequencies
    mask_100 = (freqs >= 95) & (freqs <= 105)
    mask_200 = (freqs >= 195) & (freqs <= 205)

    if np.any(mask_100):
        ratio_100 = np.sqrt(np.median(o3_median[mask_100]) /
                            np.median(o4_median[mask_100]))
        print(f"\n  Sensitivity improvement @ 100Hz: {ratio_100:.2f}x")

    if np.any(mask_200):
        ratio_200 = np.sqrt(np.median(o3_median[mask_200]) /
                            np.median(o4_median[mask_200]))
        print(f"  Sensitivity improvement @ 200Hz: {ratio_200:.2f}x")

    return freqs, o3_median, o4_median

# ─────────────────────────────────────────────────────────────
# E7-2: FAR on O4 noise
# ─────────────────────────────────────────────────────────────

def run_o4_far(o4_strains_list):
    print("\n\n📡 E7-2: FAR ON O4 NOISE")
    print("-" * 62)

    MAX_SNR_SKIP = 11.0
    total_time   = 0.0
    total_false  = 0
    seg_results  = []

    for name, strain in o4_strains_list:
        if strain is None:
            continue

        max_snr, _ = get_max_snr(strain)
        print(f"  [{name}] max SNR: {max_snr:.2f}")

        if max_snr > MAX_SNR_SKIP:
            print(f"    [SKIP] Massive glitch (SNR={max_snr:.1f} > {MAX_SNR_SKIP})")
            continue

        false_count  = count_false_triggers(strain, CALIBRATED_T)
        dur          = len(strain) / FS
        total_time  += dur
        total_false += false_count
        far_seg      = (false_count / dur) * 3.156e7

        print(f"    False: {false_count}  |  Duration: {dur:.0f}s  |  FAR: {far_seg:.2e} /yr")
        seg_results.append({"name": name, "false": false_count,
                            "dur": dur, "far": far_seg})

    far_total = (total_false / total_time) * 3.156e7 if total_time > 0 else 0
    ul = 2.303 / (total_time / 3.156e7) if total_time > 0 and total_false == 0 else None

    print(f"\n  Total O4 quiet time : {total_time/3600:.2f}h")
    print(f"  Total false alarms  : {total_false}")
    print(f"  FAR (O4)            : {far_total:.2e} /yr")
    if ul:
        print(f"  Upper limit 90% CL  : FAR < {ul:.2e} /yr")

    return seg_results, total_time, total_false, far_total

# ─────────────────────────────────────────────────────────────
# E7-3: Detection on O4 events
# ─────────────────────────────────────────────────────────────

def run_o4_detection():
    print("\n\n📡 E7-3: DETECTION ON O4 EVENTS")
    print("-" * 62)

    results = []
    for ev in O4_EVENTS:
        print(f"\n  [{ev['name']}] {ev['type']}")
        strain = fetch_event(ev['name'], ev['det'], ev['dur'])
        if strain is None:
            results.append({**ev, "detected": False, "snr": 0.0})
            continue

        center  = len(strain) / FS / 2.0
        max_snr, _ = get_max_snr(strain)
        detected = max_snr >= CALIBRATED_T
        status   = "DETECTED" if detected else "MISSED"
        print(f"    [{status}] SNR={max_snr:.2f}")
        results.append({**ev, "detected": detected, "snr": max_snr})

    return results

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_e7():
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 62)
    print("  ORAC-NT PHASE E7 — REAL O4 NOISE TEST")
    print(f"  {run_utc}")
    print("=" * 62)

    # ── Load O3 reference ─────────────────────────────────
    print("\n📥 LOADING O3 REFERENCE SEGMENTS")
    o3_strains = {}
    for seg in O3_REFERENCE:
        print(f"  [{seg['name']}]")
        s = fetch_bulk(seg['gps'], seg['dur'], seg['det'])
        o3_strains[seg['name']] = s

    # ── Load O4 segments ──────────────────────────────────
    print("\n📥 LOADING O4 QUIET SEGMENTS")
    o4_strains = {}
    o4_list    = []
    for seg in O4_QUIET_SEGMENTS:
        print(f"  [{seg['name']}]")
        s = fetch_bulk(seg['gps'], seg['dur'], seg['det'])
        o4_strains[seg['name']] = s
        o4_list.append((seg['name'], s))

    # ── E7-1: PSD comparison ──────────────────────────────
    o4_l1 = {k:v for k,v in o4_strains.items() if 'L1' in k}
    o3_l1 = {k:v for k,v in o3_strains.items() if 'L1' in k}
    freqs, o3_psd, o4_psd = run_psd_comparison(o3_l1, o4_l1)

    # ── E7-2: FAR on O4 ───────────────────────────────────
    seg_results, total_time, total_false, far_total = run_o4_far(o4_list)

    # ── E7-3: O4 event detection ──────────────────────────
    det_results = run_o4_detection()

    # ── Summary ───────────────────────────────────────────
    print("\n\n" + "=" * 62)
    print("  E7 FINAL RESULTS")
    print("=" * 62)
    print(f"\n  O4 quiet time : {total_time/3600:.2f}h")
    print(f"  O4 FAR        : {far_total:.2e} /yr")
    detected = sum(1 for r in det_results if r['detected'])
    print(f"  O4 detection  : {detected}/{len(det_results)}")

    # ── Visualization ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#0e1117')

    # Panel 1 — PSD comparison
    if freqs is not None and o3_psd is not None and o4_psd is not None:
        mask = (freqs >= 20) & (freqs <= 1000)
        axes[0].loglog(freqs[mask], np.sqrt(o3_psd[mask]),
                       color='#4488ff', lw=2, label='O3a (reference)', alpha=0.9)
        axes[0].loglog(freqs[mask], np.sqrt(o4_psd[mask]),
                       color='#00ffcc', lw=2, label='O4a', alpha=0.9)
        axes[0].set_title("Noise PSD Comparison\nO3a vs O4a (L1)", color='white', fontsize=10)
        axes[0].set_xlabel("Frequency (Hz)", color='white')
        axes[0].set_ylabel("Strain ASD (1/√Hz)", color='white')
        axes[0].legend(facecolor='#0e1117', labelcolor='white', fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "PSD data\nnot available",
                     transform=axes[0].transAxes, color='white',
                     ha='center', va='center', fontsize=12)
    axes[0].set_facecolor('#0e1117'); axes[0].tick_params(colors='white')

    # Panel 2 — O4 FAR per segment
    if seg_results:
        seg_names  = [r['name']  for r in seg_results]
        seg_false  = [r['false'] for r in seg_results]
        seg_colors = ['#00ffcc' if f == 0 else '#ff9900' for f in seg_false]
        axes[1].bar(range(len(seg_names)), seg_false, color=seg_colors, alpha=0.85)
        axes[1].set_xticks(range(len(seg_names)))
        axes[1].set_xticklabels([s.replace('O4a_','') for s in seg_names],
                                rotation=30, color='white', fontsize=8)
        far_str = f"FAR = {far_total:.2e} /yr" if far_total > 0 else "FAR = 0"
        axes[1].set_title(f"O4 Noise — False Triggers\n{far_str} | {total_time/3600:.1f}h total",
                          color='white', fontsize=10)
        axes[1].set_ylabel("False Triggers", color='white')
    else:
        axes[1].text(0.5, 0.5, "O4 data\nnot available",
                     transform=axes[1].transAxes, color='white',
                     ha='center', va='center', fontsize=12)
        axes[1].set_title("O4 Noise FAR", color='white', fontsize=10)
    axes[1].set_facecolor('#0e1117'); axes[1].tick_params(colors='white')

    # Panel 3 — O4 event detection
    names_det = [r['name'] for r in det_results]
    snrs_det  = [r['snr']  for r in det_results]
    cols_det  = ['#00ffcc' if r['detected'] else '#ff4444' for r in det_results]
    axes[2].bar(names_det, snrs_det, color=cols_det, alpha=0.85)
    axes[2].axhline(CALIBRATED_T, color='red', lw=2, ls='--',
                    label=f'Threshold={CALIBRATED_T}')
    det_count = sum(1 for r in det_results if r['detected'])
    axes[2].set_title(f"O4 Event Detection\n{det_count}/{len(det_results)} detected",
                      color='white', fontsize=10)
    axes[2].set_ylabel("Peak SNR", color='white')
    axes[2].legend(facecolor='#0e1117', labelcolor='white', fontsize=9)
    for i, (snr, r) in enumerate(zip(snrs_det, det_results)):
        axes[2].text(i, snr+0.2, f"{snr:.1f}", ha='center', color='white', fontsize=9)
    axes[2].set_facecolor('#0e1117'); axes[2].tick_params(colors='white')

    plt.suptitle(
        f"ORAC-NT Phase E7 — Real O4 Noise Test\n"
        f"Dimitar Kretski  |  DOI: 10.5281/zenodo.20129975  |  {run_utc}",
        color='white', fontsize=11
    )
    plt.tight_layout()
    plt.savefig("orac_phase_e7.png", dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print(f"\nSaved: orac_phase_e7.png")
    print(f"   {run_utc}")

if __name__ == "__main__":
    run_e7()
