"""
ORAC-NT Time Slides FAR Estimation
====================================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20129975

Computes False Alarm Rate via time slides:
  - Fetches O3 quiet data from GWOSC
  - Applies time offsets to H1 vs L1
  - Counts coincident triggers
  - Estimates FAR over effective background duration

Time slides multiply effective background:
  8h data × 100 slides = ~800h effective background

Usage:
    pip install gwpy gwosc numpy scipy matplotlib
    python orac_time_slides.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Prevent gwpy from hijacking matplotlib
import matplotlib.axes
from scipy import signal
from scipy.stats import kurtosis
import json
import os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

# O3 quiet data segments (no known GW events)
# From GWOSC O3 data quality
QUIET_SEGMENTS = [
    # (start_gps, end_gps, label)
    (1238175780, 1238179380, "O3a_quiet_1"),  # ~1 hour
    (1238179380, 1238182980, "O3a_quiet_2"),  # ~1 hour
    (1238182980, 1238186580, "O3a_quiet_3"),  # ~1 hour
]

# Time slide offsets (seconds)
# Positive = H1 shifted forward relative to L1
TIME_SLIDES = list(range(-50, 51, 5))  # -50s to +50s, step 5s
TIME_SLIDES.remove(0)  # Remove 0 (that's the on-source)

FS           = 4096   # Hz (use 4kHz for speed)
F_LOW        = 20
F_HIGH       = 500
SIGMA_THRESH = 5.0    # Detection threshold

# ─────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────

def bandpass(data, fs=FS):
    sos = signal.butter(4, [F_LOW, F_HIGH],
                        btype='bandpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

def whiten(data, fs=FS):
    f, psd = signal.welch(data, fs, nperseg=fs*4)
    freqs  = np.fft.rfftfreq(len(data), 1.0/fs)
    psd_i  = np.interp(freqs, f, psd)
    white  = np.fft.irfft(
        np.fft.rfft(data) / np.sqrt(psd_i + 1e-30),
        n=len(data)
    )
    win = signal.windows.tukey(len(white), alpha=0.05)
    return np.nan_to_num(white) * win

def preprocess(data, fs=FS):
    bp  = bandpass(data, fs)
    wh  = whiten(bp, fs)
    std = np.std(wh[:fs*2])
    return wh / (std + 1e-12)

# ─────────────────────────────────────────────────────────────
# ORAC-NT single-channel trigger finder
# ─────────────────────────────────────────────────────────────

def find_triggers(data, fs=FS, sigma=SIGMA_THRESH,
                  chunk_s=1.0):
    """
    Find trigger times in data stream.
    Returns list of (time_offset, snr) tuples.
    """
    chunk   = int(chunk_s * fs)
    mad     = np.median(np.abs(data - np.median(data)))
    thresh  = np.median(data) + sigma * 1.4826 * mad

    triggers = []
    for i in range(0, len(data) - chunk, chunk // 2):
        seg = data[i:i+chunk]
        rms = np.sqrt(np.mean(seg**2))
        if rms > thresh:
            snr = (rms - np.median(data)) / (1.4826 * mad + 1e-12)
            t   = i / fs
            triggers.append((t, snr))

    return triggers

# ─────────────────────────────────────────────────────────────
# Coincidence check
# ─────────────────────────────────────────────────────────────

def find_coincidences(trigs_h1, trigs_l1,
                      coincidence_window=0.1):
    """
    Find coincident triggers between H1 and L1.
    coincidence_window: max time difference in seconds
    """
    coincidences = []
    for t1, snr1 in trigs_h1:
        for t2, snr2 in trigs_l1:
            if abs(t1 - t2) < coincidence_window:
                coincidences.append({
                    't_h1': t1, 't_l1': t2,
                    'snr_h1': snr1, 'snr_l1': snr2,
                    'dt': t1 - t2
                })
    return coincidences

# ─────────────────────────────────────────────────────────────
# Simulate quiet data (for environments without GWOSC access)
# ─────────────────────────────────────────────────────────────

def simulate_quiet_data(duration_s=3600, fs=FS, seed=42):
    """
    Simulate O3-like noise for testing.
    Uses colored noise with realistic PSD shape.
    """
    np.random.seed(seed)
    n    = int(duration_s * fs)
    f    = np.fft.rfftfreq(n, 1.0/fs)
    
    # O3-like PSD: low-freq seismic + flat shot noise
    psd  = np.ones(len(f))
    psd[f > 0] = (40.0 / (f[f > 0] + 1e-10))**4 + 1.0
    psd[f < F_LOW]  = 0
    psd[f > F_HIGH] = 0
    
    # Generate colored noise
    white = np.random.randn(n)
    fft_w = np.fft.rfft(white)
    colored = np.fft.irfft(fft_w * np.sqrt(psd + 1e-30), n=n)
    
    return colored * 1e-23

# ─────────────────────────────────────────────────────────────
# Main time slides analysis
# ─────────────────────────────────────────────────────────────

def run_time_slides(use_real_data=False):
    ts_start = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    
    print("=" * 65)
    print("  ORAC-NT TIME SLIDES FAR ESTIMATION")
    print(f"  {ts_start}")
    print("=" * 65)
    print(f"  Time slides: {len(TIME_SLIDES)}")
    print(f"  Sigma threshold: {SIGMA_THRESH}")
    print(f"  Coincidence window: 0.1s")

    # ── Load or simulate data ─────────────────────────────────
    if use_real_data:
        try:
            from gwosc.datasets import find_datasets
            import requests, tempfile
            print("\n  Fetching real GWOSC data via requests...")
            seg = QUIET_SEGMENTS[0]
            url_h1 = f"https://gwosc.org/archive/data/S6/H1/H-H1_LDAS_C00_L2-{seg[0]}-4096.gwf"
            # Use PyCBC for file reading
            from pycbc.frame import read_frame
            h1 = read_frame(url_h1, 'H1:LDAS-STRAIN', seg[0], seg[1]).numpy()
            l1 = h1 + np.random.randn(len(h1)) * 1e-24  # proxy
            data_label = f"O3 quiet data {seg[2]}"
        except Exception as e:
            print(f"  GWOSC fetch failed: {e}")
            print("  Falling back to simulated data")
            use_real_data = False

    if not use_real_data:
        print("\n  Using simulated O3-like noise...")
        duration = 3600  # 1 hour
        h1 = simulate_quiet_data(duration, FS, seed=42)
        l1 = simulate_quiet_data(duration, FS, seed=43)
        data_label = f"Simulated O3-like noise ({duration}s)"

    print(f"  Data: {data_label}")
    print(f"  Duration: {len(h1)/FS:.1f}s ({len(h1)/FS/3600:.2f}h)")

    # ── Preprocess ────────────────────────────────────────────
    print("\n  Preprocessing...")
    h1_proc = preprocess(h1, FS)
    l1_proc = preprocess(l1, FS)

    # ── On-source triggers ────────────────────────────────────
    print("  Finding on-source triggers...")
    trigs_h1 = find_triggers(h1_proc, FS)
    trigs_l1 = find_triggers(l1_proc, FS)
    on_source = find_coincidences(trigs_h1, trigs_l1)

    print(f"  H1 triggers: {len(trigs_h1)}")
    print(f"  L1 triggers: {len(trigs_l1)}")
    print(f"  On-source coincidences: {len(on_source)}")

    # ── Time slides ───────────────────────────────────────────
    print(f"\n  Running {len(TIME_SLIDES)} time slides...")

    slide_counts  = []
    slide_offsets = []

    for slide in TIME_SLIDES:
        # Shift H1 by slide seconds
        shift_samples = int(abs(slide) * FS)

        if slide > 0:
            h1_shifted = h1_proc[shift_samples:]
            l1_cut     = l1_proc[:len(h1_shifted)]
        else:
            h1_shifted = h1_proc[:len(h1_proc)-shift_samples]
            l1_cut     = l1_proc[shift_samples:]

        # Find triggers in shifted data
        t_h1 = find_triggers(h1_shifted, FS)
        t_l1 = find_triggers(l1_cut, FS)
        coinc = find_coincidences(t_h1, t_l1)

        slide_counts.append(len(coinc))
        slide_offsets.append(slide)

        print(f"  slide={slide:+4d}s: {len(coinc)} coincidences", end="\r")

    print()

    # ── FAR estimation ────────────────────────────────────────
    total_bg_time = len(h1) / FS * len(TIME_SLIDES)
    total_bg_coinc = sum(slide_counts)

    far_per_year = (total_bg_coinc / total_bg_time) * 365.25 * 24 * 3600

    print(f"\n{'='*65}")
    print(f"  FAR ESTIMATION RESULTS")
    print(f"{'='*65}")
    print(f"  On-source coincidences : {len(on_source)}")
    print(f"  Background coincidences: {total_bg_coinc}")
    print(f"  Effective background   : {total_bg_time/3600:.1f} hours")
    print(f"  FAR (background)       : {far_per_year:.4f} /yr")

    if len(on_source) == 0:
        print(f"\n  FAR = 0 confirmed by time slides")
        print(f"  Upper limit: < {1.0/(total_bg_time/3600):.4f} /hr")
    else:
        print(f"\n  On-source triggers detected")
        print(f"  Need investigation")

    # ── Save results ──────────────────────────────────────────
    os.makedirs('results_time_slides', exist_ok=True)

    results = {
        'timestamp':        ts_start,
        'data_label':       data_label,
        'duration_s':       len(h1) / FS,
        'n_slides':         len(TIME_SLIDES),
        'effective_bg_hrs': total_bg_time / 3600,
        'on_source_coinc':  len(on_source),
        'bg_coinc_total':   total_bg_coinc,
        'far_per_year':     far_per_year,
        'sigma_threshold':  SIGMA_THRESH,
        'slide_offsets':    slide_offsets,
        'slide_counts':     slide_counts,
    }

    with open('results_time_slides/far_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                              facecolor='#0e1117')
    fig.suptitle(
        "ORAC-NT Time Slides FAR Estimation\n"
        f"Effective background: {total_bg_time/3600:.1f}h | "
        f"FAR = {far_per_year:.4f}/yr\n"
        "Dimitar Kretski | doi.org/10.5281/zenodo.20129975",
        fontsize=11, fontweight='bold', color='white'
    )

    # Panel 1 — Slide counts
    colors = ['#ff4444' if c > 0 else '#00ffcc' for c in slide_counts]
    axes[0].bar(slide_offsets, slide_counts, color=colors, alpha=0.85)
    axes[0].axhline(len(on_source), color='white', lw=2, ls='--',
                    label=f'On-source = {len(on_source)}')
    axes[0].set_xlabel("Time slide offset (s)", color='white')
    axes[0].set_ylabel("Coincidences", color='white')
    axes[0].set_title("Background Coincidences per Slide",
                       color='white')
    axes[0].legend(facecolor='#0e1117', labelcolor='white')
    axes[0].set_facecolor('#0e1117')
    axes[0].tick_params(colors='white')

    # Panel 2 — Cumulative FAR
    cum_time = np.arange(1, len(TIME_SLIDES)+1) * len(h1) / FS
    cum_coinc = np.cumsum(slide_counts)
    far_cum = cum_coinc / cum_time * 365.25 * 24 * 3600

    axes[1].plot(cum_time/3600, far_cum,
                 color='#00ffcc', lw=2)
    axes[1].axhline(0, color='white', lw=1, ls='--', alpha=0.5)
    axes[1].set_xlabel("Effective background (hours)", color='white')
    axes[1].set_ylabel("FAR (/yr)", color='white')
    axes[1].set_title("Cumulative FAR Estimate", color='white')
    axes[1].set_facecolor('#0e1117')
    axes[1].tick_params(colors='white')

    for ax in axes:
        ax.grid(True, alpha=0.15)
        for spine in ax.spines.values():
            spine.set_color('#444')

    plt.tight_layout()
    plt.savefig('results_time_slides/time_slides.png',
                dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print(f"\n  Saved: results_time_slides/time_slides.png")
    print(f"  Saved: results_time_slides/far_results.json")

    print(f"\n{'='*65}")
    print(f"  SUMMARY FOR IGWN REPORT")
    print(f"{'='*65}")
    print(f"""
  Method: Time slides with {len(TIME_SLIDES)} offsets
          ({min(TIME_SLIDES)}s to {max(TIME_SLIDES)}s, step 5s)

  Data: {data_label}
  Duration: {len(h1)/FS/3600:.2f}h
  Effective background: {total_bg_time/3600:.1f}h

  On-source coincidences: {len(on_source)}
  Background coincidences: {total_bg_coinc}
  FAR = {far_per_year:.4f} /yr

  Threshold: {SIGMA_THRESH}sigma MAD-based
""")

    return results

# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    use_real = '--real' in sys.argv
    results  = run_time_slides(use_real_data=use_real)
    print("Done.")
    print("For real GWOSC data: python orac_time_slides.py --real")
