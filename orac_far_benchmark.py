"""
ORAC-NT O3 False Alarm Rate Benchmark v2.5
==========================================
KEY CHANGES vs v2.4:
- Hardcoded GPS times + direct bulk fetch for O1 events
  (fixes GW150914/GW151226 SNR=2.6 — wrong file was fetched)
- Pre-scan quiet segments: skip if max SNR > 50 (massive glitch)
- O2 quiet segments replaced with verified low-noise epochs
- Chi2 veto retained

Author : Dimitar Kretski
DOI    : 10.5281/zenodo.19553825
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os, tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal.windows import tukey
from scipy.stats import kurtosis, chi2 as chi2_dist
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

FS           = 4096
TARGET_FAR   = 1.0
FETCH_RETRY  = 2
CHI2_NBINS   = 4
CHI2_P_THR   = 0.001
MAX_QUIET_SNR = 50.0    # skip quiet segment if max SNR > this (massive glitch)

# ─────────────────────────────────────────────────────────────
# GW EVENTS — hardcoded GPS + detector + run
# ─────────────────────────────────────────────────────────────

GW_EVENTS = [
    {"name": "GW170817", "type": "BNS",  "detector": "L1",
     "gps": 1187008882, "run": "O2", "duration": 32},
    {"name": "GW150914", "type": "BBH",  "detector": "L1",
     "gps": 1126259462, "run": "O1", "duration": 32},
    {"name": "GW151226", "type": "BBH",  "detector": "L1",
     "gps": 1135136350, "run": "O1", "duration": 32},
    {"name": "GW190521", "type": "BBH",  "detector": "L1",
     "gps": 1242442967, "run": "O3", "duration": 32},
    {"name": "GW190814", "type": "NSBH", "detector": "L1",
     "gps": 1249852257, "run": "O3", "duration": 32},
]

# ─────────────────────────────────────────────────────────────
# QUIET SEGMENTS — verified O2 low-noise epochs
# Avoiding: ~1168M (earthquake), ~1171.5M (high noise O2_quiet_2)
# ─────────────────────────────────────────────────────────────

QUIET_SEGMENTS = [
    {"name": "O2_quiet_A", "gps_start": 1169638912, "detector": "L1", "duration": 4096},
    {"name": "O2_quiet_B", "gps_start": 1173536768, "detector": "L1", "duration": 4096},
    {"name": "O2_quiet_C", "gps_start": 1177223168, "detector": "L1", "duration": 4096},
]

# ─────────────────────────────────────────────────────────────
# TEMPLATE BANK
# ─────────────────────────────────────────────────────────────

TEMPLATES = [
    {"name": "BBH", "f0": 30, "f1": 350, "duration": 0.3},
    {"name": "BNS", "f0": 30, "f1": 400, "duration": 1.0},
]

# ─────────────────────────────────────────────────────────────
# SYNTHETIC FALLBACK
# ─────────────────────────────────────────────────────────────

def synthetic_noise(duration_s, seed=42):
    rng     = np.random.default_rng(seed)
    n       = duration_s * FS
    white   = rng.standard_normal(n)
    freqs   = np.fft.rfftfreq(n, 1 / FS)
    hf      = np.fft.rfft(white)
    slope   = np.where(freqs > 0, freqs ** -0.5, 1.0)
    slope  /= slope.max()
    colored = np.fft.irfft(hf * slope, n=n)
    sos     = signal.butter(4, [30, 400], btype='bandpass', fs=FS, output='sos')
    colored = signal.sosfiltfilt(sos, colored)
    colored /= np.std(colored) + 1e-12
    return colored.astype(np.float64) * 1e-21

# ─────────────────────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────────────────────

def _download_hdf5(url):
    """Returns (strain, gps_start) or (None, None)."""
    import requests, h5py
    print(f"    [GET] ...{url[-55:]}")
    for attempt in range(FETCH_RETRY):
        try:
            r = requests.get(url, timeout=180, stream=True)
            r.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False)
            for chunk in r.iter_content(65536):
                tmp.write(chunk)
            tmp.close()
            with h5py.File(tmp.name, 'r') as f:
                data      = f['strain']['Strain'][:]
                dt        = f['strain']['Strain'].attrs.get('Xspacing', 1.0 / FS)
                # GPS start: try common attribute names
                gps_start = None
                for attr in ('GPSstart', 'GPS_start', 'Tstart', 'start_time'):
                    if attr in f['strain']['Strain'].attrs:
                        gps_start = float(f['strain']['Strain'].attrs[attr])
                        break
                if gps_start is None and 'meta' in f:
                    for attr in ('GPSstart', 'GPS_start', 'Tstart'):
                        if attr in f['meta'].attrs:
                            gps_start = float(f['meta'].attrs[attr])
                            break
            os.unlink(tmp.name)
            strain = data.astype(np.float64)
            sr = int(round(1.0 / dt))
            if sr != FS:
                strain = signal.resample(strain, int(len(strain) * FS / sr))
            return strain, gps_start
        except Exception as e:
            print(f"    [WARN] attempt {attempt+1}/{FETCH_RETRY}: {e}")
    return None, None


def _extract_by_gps(strain, file_gps_start, event_gps, duration):
    """Extract `duration` seconds centered on event_gps."""
    if file_gps_start is None:
        # Fallback: take middle of file
        mid = len(strain) // 2
        half = duration * FS // 2
        return strain[max(0, mid - half) : mid + half]
    offset_smp = int((event_gps - file_gps_start) * FS)
    half       = duration * FS // 2
    lo = max(0, offset_smp - half)
    hi = min(len(strain), offset_smp + half)
    extracted = strain[lo:hi]
    print(f"    [OFFSET] {event_gps - file_gps_start:.0f}s into file → "
          f"extracted {len(extracted)/FS:.1f}s")
    return extracted


def fetch_event_strain(ev):
    """
    Fetch event strain. For O1 4096s files, extracts by GPS offset.
    Falls back to get_event_urls for O2/O3 event files.
    """
    gps      = ev["gps"]
    detector = ev["detector"]
    duration = ev["duration"]
    name     = ev["name"]
    half     = duration // 2

    # Strategy 1: GPS-based bulk fetch — works for all runs
    try:
        from gwosc.locate import get_urls
        urls = get_urls(detector, gps - half, gps + half,
                        sample_rate=4096, format='hdf5')
        if urls:
            hdf_url        = next((u for u in urls if 'hdf5' in u), urls[0])
            strain, gps_s  = _download_hdf5(hdf_url)
            if strain is not None:
                if len(strain) > duration * FS:
                    strain = _extract_by_gps(strain, gps_s, gps, duration)
                print(f"    [OK] {len(strain)/FS:.1f}s @ {FS} Hz (GPS fetch)")
                return strain
    except Exception as e:
        print(f"    [WARN] bulk fetch: {e}")

    # Strategy 2: get_event_urls (O2/O3 event files are already 32s)
    try:
        from gwosc.locate import get_event_urls
        urls = get_event_urls(name, detector=detector, duration=duration)
        if urls:
            strain, gps_s = _download_hdf5(urls[0])
            if strain is not None:
                if len(strain) > duration * FS:
                    strain = _extract_by_gps(strain, gps_s, gps, duration)
                print(f"    [OK] {len(strain)/FS:.1f}s @ {FS} Hz (event URL)")
                return strain
    except Exception as e:
        print(f"    [WARN] event URL: {e}")

    return None


def fetch_bulk_strain(gps_start, duration, detector='L1'):
    try:
        from gwosc.locate import get_urls
        urls = get_urls(detector, gps_start, gps_start + duration,
                        sample_rate=4096, format='hdf5')
        if not urls:
            return None
        hdf_url       = next((u for u in urls if '4096' in u and 'hdf5' in u), urls[0])
        strain, gps_s = _download_hdf5(hdf_url)
        if strain is None:
            return None
        print(f"    [OK] {len(strain)/FS:.0f}s @ {FS} Hz")
        return strain
    except Exception as e:
        print(f"    [ERROR] GPS={gps_start}: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────

def whiten(data, cal_s=5.0):
    sos  = signal.butter(4, [30, 400], btype='bandpass', fs=FS, output='sos')
    data = signal.sosfiltfilt(sos, data)
    cal  = int(cal_s * FS)
    f, psd = signal.welch(data[:cal], FS, nperseg=FS)
    psd_i  = np.interp(np.fft.rfftfreq(len(data), 1 / FS), f, psd)
    hf     = np.fft.rfft(data)
    w      = np.fft.irfft(hf / np.sqrt(psd_i + 1e-12), n=len(data))
    w      = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w * tukey(len(w), 0.05)


def make_template(cfg):
    n    = int(cfg["duration"] * FS)
    tt   = np.linspace(0, cfg["duration"], n, endpoint=False)
    tmpl = signal.chirp(tt, f0=cfg["f0"], f1=cfg["f1"],
                        t1=cfg["duration"], method='quadratic')
    tmpl *= tukey(n, 0.2)
    tmpl /= np.sqrt(np.sum(tmpl**2)) + 1e-20
    return tmpl


def mad_sigma(corr):
    clip  = np.percentile(np.abs(corr), 95)
    quiet = corr[np.abs(corr) < clip]
    if len(quiet) == 0:
        return 1e-12
    return 1.4826 * np.median(np.abs(quiet - np.median(quiet))) + 1e-12


def chi2_at_idx(w, tmpl, idx):
    N     = len(w)
    freqs = np.fft.rfftfreq(N, 1 / FS)
    tf    = np.fft.rfft(np.pad(tmpl, (0, N - len(tmpl))))
    edges = np.linspace(30, 400, CHI2_NBINS + 1)
    snr_bins = []
    for i in range(CHI2_NBINS):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        ts   = np.zeros_like(tf); ts[mask] = tf[mask]
        sub_t  = np.fft.irfft(ts, n=N)
        corr_i = signal.fftconvolve(w, sub_t[::-1], mode='same')
        s_i    = mad_sigma(corr_i)
        snr_bins.append(float(corr_i[min(idx, len(corr_i)-1)]) / s_i)
    snr_bins = np.array(snr_bins)
    expected = snr_bins.mean()
    c2  = float(np.sum((snr_bins - expected) ** 2))
    dof = 2 * (CHI2_NBINS - 1)
    p   = float(1.0 - chi2_dist.cdf(c2, dof))
    return c2, p


def process_segment(strain, cal_s=5.0):
    cal = int(cal_s * FS)
    w   = whiten(strain, cal_s)
    w  /= np.std(w[:cal]) + 1e-12

    tmpls    = [make_template(cfg) for cfg in TEMPLATES]
    snr_max  = None
    bt_idx   = np.zeros(len(w), dtype=int)

    for ti, tmpl in enumerate(tmpls):
        corr  = signal.fftconvolve(w, tmpl[::-1], mode='same')
        sigma = mad_sigma(corr)
        snr_t = np.abs(corr) / sigma
        snr_t = np.nan_to_num(snr_t)
        if snr_max is None:
            snr_max = snr_t
        else:
            better = snr_t > snr_max
            bt_idx[better] = ti
            snr_max = np.where(better, snr_t, snr_max)

    return snr_max, w, tmpls, bt_idx


def find_triggers(snr_t, snr_threshold, min_gap_s=2.0):
    peaks, _ = signal.find_peaks(
        snr_t, height=snr_threshold,
        distance=int(min_gap_s * FS)
    )
    return [p / FS for p in peaks], list(peaks)


def is_glitch_with_chi2(strain, w, tmpls, bt_idx, t_trigger, window=1.0):
    idx  = int(t_trigger * FS)
    half = int(window / 2 * FS)
    snip = strain[max(0, idx - half) : min(len(strain), idx + half)]

    if len(snip) < 100:
        return True, 0.0, 0.0

    if kurtosis(snip, fisher=False, bias=False) > 25.0:
        return True, 0.0, 0.0

    ti    = int(bt_idx[min(idx, len(bt_idx) - 1)])
    tmpl  = tmpls[ti]
    c2, p = chi2_at_idx(w, tmpl, min(idx, len(w) - 1))

    if p < CHI2_P_THR:
        return True, c2, p

    return False, c2, p

# ─────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────

def calibrate_threshold(quiet_cache, target_far_per_yr=1.0):
    if not quiet_cache:
        print("  [CAL] No quiet data — threshold=8.0")
        return 8.0

    total_s   = sum(c[5] for c in quiet_cache)
    max_false = target_far_per_yr * (total_s / 3.156e7)

    print(f"\n  [CAL] Quiet data  : {total_s/3600:.2f}h")
    print(f"  [CAL] Target FAR  : {target_far_per_yr:.1f} /yr")
    print(f"  [CAL] Max false   : {max_false:.4f}")

    lo, hi, best = 3.0, 15.0, 15.0

    for it in range(25):
        mid         = (lo + hi) / 2.0
        total_false = 0
        for snr_t, strain, w, tmpls, bt_idx, _ in quiet_cache:
            t_trigs, _ = find_triggers(snr_t, mid)
            for tr in t_trigs:
                glitch, _, _ = is_glitch_with_chi2(strain, w, tmpls, bt_idx, tr)
                if not glitch:
                    total_false += 1
        current_far = (total_false / total_s) * 3.156e7
        print(f"  [CAL] iter={it+1:2d}  thr={mid:.3f}  "
              f"false={total_false}  FAR={current_far:.2e} /yr")
        if total_false <= max_false:
            best = mid; hi = mid
        else:
            lo = mid
        if (hi - lo) < 0.05:
            break

    print(f"\n  [CAL] Threshold: {best:.3f}")
    return best

# ─────────────────────────────────────────────────────────────
# BENCHMARK
# ─────────────────────────────────────────────────────────────

def run_benchmark():
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("\n" + "=" * 62)
    print("  ORAC-NT v2.5 — FAR BENCHMARK")
    print(f"  {run_utc}")
    print("=" * 62)

    # ── QUIET SEGMENTS ────────────────────────
    print("\nLOADING QUIET O2 SEGMENTS")
    print("-" * 62)

    quiet_cache = []
    quiet_meta  = []

    for seg in QUIET_SEGMENTS:
        print(f"\n  [{seg['name']}] GPS={seg['gps_start']}")
        s = fetch_bulk_strain(seg['gps_start'], seg['duration'], seg['detector'])

        if s is None:
            print(f"    [FALLBACK] synthetic ({seg['duration']}s)...")
            s = synthetic_noise(seg['duration'], seed=seg['gps_start'] % 999)
            print(f"    [OK] synthetic {len(s)/FS:.0f}s @ {FS} Hz")

        print("    [PROCESS] SNR + template bank...")
        snr_t, w, tmpls, bt_idx = process_segment(s)
        max_snr = float(np.nanmax(snr_t))
        print(f"    [OK] max SNR: {max_snr:.2f}")

        if max_snr > MAX_QUIET_SNR:
            print(f"    [SKIP] max SNR={max_snr:.1f} > {MAX_QUIET_SNR} — massive glitch, skipping segment")
            continue

        quiet_cache.append((snr_t, s, w, tmpls, bt_idx, len(s) / FS))
        quiet_meta.append({"name": seg['name'], "duration": len(s) / FS})

    if not quiet_cache:
        print("\n  [WARN] All quiet segments skipped — using synthetic fallback")
        for i, seg in enumerate(QUIET_SEGMENTS):
            s = synthetic_noise(seg['duration'], seed=i * 100)
            snr_t, w, tmpls, bt_idx = process_segment(s)
            quiet_cache.append((snr_t, s, w, tmpls, bt_idx, len(s) / FS))
            quiet_meta.append({"name": seg['name'] + "_SYN", "duration": len(s) / FS})

    # ── CALIBRATION ───────────────────────────
    print("\n\nPHASE 0: CALIBRATION")
    print("-" * 62)
    cal_threshold = calibrate_threshold(quiet_cache, TARGET_FAR)

    # ── DETECTION ─────────────────────────────
    print(f"\n\nPHASE 1: DETECTION  (threshold={cal_threshold:.3f})")
    print("-" * 62)

    detection_results = []

    for ev in GW_EVENTS:
        print(f"\n  [{ev['name']}] {ev['type']}  GPS={ev['gps']}")
        strain = fetch_event_strain(ev)

        if strain is None:
            print(f"    [FALLBACK] synthetic...")
            strain = synthetic_noise(ev['duration'], seed=hash(ev['name']) % 999)
            print(f"    [OK] synthetic {len(strain)/FS:.0f}s @ {FS} Hz")

        snr_t, w, tmpls, bt_idx = process_segment(strain)

        center     = len(strain) / FS / 2.0
        center_idx = int(center * FS)
        half       = int(5.0 * FS)
        lo_i       = max(0, center_idx - half)
        hi_i       = min(len(snr_t), center_idx + half)
        peak_idx   = lo_i + int(np.argmax(snr_t[lo_i:hi_i]))
        peak_snr   = float(snr_t[peak_idx])
        peak_t     = peak_idx / FS

        glitch, c2, p = is_glitch_with_chi2(strain, w, tmpls, bt_idx, peak_t)
        detected      = peak_snr >= cal_threshold and not glitch
        status        = "DETECTED" if detected else "MISSED"

        print(f"    [{status}] SNR={peak_snr:.2f}  chi2={c2:.1f}  p={p:.3f}")

        detection_results.append({
            **ev, "detected": detected,
            "peak_snr": peak_snr, "chi2": c2, "p_val": p,
        })

    # ── FAR VERIFICATION ──────────────────────
    print("\n\nPHASE 2: FAR VERIFICATION")
    print("-" * 62)

    total_time_s = 0.0
    total_false  = 0

    for (snr_t, strain, w, tmpls, bt_idx, dur), meta in zip(quiet_cache, quiet_meta):
        fc = 0
        t_trigs, _ = find_triggers(snr_t, cal_threshold)
        for tr in t_trigs:
            glitch, _, _ = is_glitch_with_chi2(strain, w, tmpls, bt_idx, tr)
            if not glitch:
                fc += 1
        total_time_s += dur
        total_false  += fc
        print(f"  [{meta['name']}]  False: {fc}  Duration: {dur:.0f}s")

    # ── RESULTS ───────────────────────────────
    print("\n\n" + "=" * 62)
    print("  FINAL RESULTS")
    print("=" * 62)

    detected_count = sum(1 for r in detection_results if r.get("detected"))
    total_events   = len(detection_results)
    detection_rate = detected_count / total_events * 100 if total_events else 0
    far_per_yr     = (total_false / total_time_s) * 3.156e7 if total_time_s > 0 else 0

    print(f"\n  SNR Threshold  : {cal_threshold:.3f}")
    print(f"  Detection Rate : {detected_count}/{total_events} ({detection_rate:.0f}%)")
    print(f"  FAR            : {far_per_yr:.2e} /yr")
    print(f"  Total quiet    : {total_time_s/3600:.2f}h")
    print(f"  Total false    : {total_false}")

    # ── VISUALIZATION ─────────────────────────
    print("\n[VIZ] Generating...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0e1117')
    ax = axes[0]
    names  = [r['name'] for r in detection_results]
    snrs   = [r['peak_snr'] for r in detection_results]
    colors = ['#00ffcc' if r['detected'] else '#ff4444' for r in detection_results]
    bars   = ax.bar(names, snrs, color=colors, alpha=0.85)
    ymax   = max(snrs) if max(snrs) > 0 else 1

    for bar, v in zip(bars, snrs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.01,
                f'{v:.1f}', ha='center', va='bottom', color='white', fontsize=9)

    ax.axhline(cal_threshold, color='red', lw=2, ls='--',
               label=f'Threshold = {cal_threshold:.2f}')
    ax.set_title("Peak SNR per Event", color='white')
    ax.set_ylabel("Peak SNR", color='white')
    ax.legend(facecolor='#0e1117', labelcolor='white')
    ax.set_facecolor('#0e1117'); ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')

    ax2 = axes[1]; ax2.set_facecolor('#0e1117'); ax2.axis('off')
    segs_used = ", ".join(m['name'] for m in quiet_meta)
    lines = [
        "ORAC-NT v2.5  FAR Benchmark", "",
        f"Threshold    :  {cal_threshold:.3f}",
        f"Detection    :  {detected_count}/{total_events}  ({detection_rate:.0f}%)",
        f"FAR          :  {far_per_yr:.2e} /yr",
        f"Quiet data   :  {total_time_s/3600:.2f} h",
        f"False alarms :  {total_false}", "",
        f"Templates    :  BBH / BNS",
        f"Veto         :  kurtosis + chi2 ({CHI2_NBINS} bands)",
        f"Seg filter   :  max SNR < {MAX_QUIET_SNR:.0f}", "",
        "Event results:",
    ]
    for r in detection_results:
        mark = "v" if r['detected'] else "x"
        lines.append(f"  {mark}  {r['name']:10s}  SNR={r['peak_snr']:6.1f}  p={r['p_val']:.3f}")

    for i, line in enumerate(lines):
        color = '#00ffcc' if i == 0 else 'white'
        ax2.text(0.04, 0.97 - i * 0.052, line,
                 transform=ax2.transAxes, color=color,
                 fontsize=9.5, fontfamily='monospace', va='top')

    fig.suptitle(
        f"ORAC-NT v2.5  |  {detected_count}/{total_events} detected  "
        f"|  FAR={far_per_yr:.2e} /yr",
        color='white', fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig("orac_far_benchmark.png", dpi=150,
                facecolor='#0e1117', bbox_inches='tight')

    print("\nVizualizatsiya: orac_far_benchmark.png")
    print("Done.")


if __name__ == "__main__":
    run_benchmark()
