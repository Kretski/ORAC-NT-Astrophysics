"""
ORAC-NT Phase C Benchmark
==========================
Author : Dimitar Kretski
DOI    : 10.5281/zenodo.20098932

1. MULTI-DETECTOR COINCIDENCE  — H1 + L1 coincidence window Δt < 10ms
2. GLITCH REJECTION BENCHMARK  — blip, scattered light, burst glitches
3. LATENCY METRICS             — conditioning, trigger, MF confirmation
4. ROC with coincidence        — AUC improvement

ИНСТАЛАЦИЯ:
  pip install gwosc numpy scipy matplotlib requests h5py

УПОТРЕБА:
  python orac_phase_c.py
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
COINCIDENCE_WINDOW = 0.010   # 10ms — light travel time LIGO H1-L1

# ─────────────────────────────────────────────────────────────
# GW Events — с двата детектора
# ─────────────────────────────────────────────────────────────

GW_EVENTS_COINCIDENCE = [
    {"name": "GW170817", "type": "BNS",  "duration": 32},
    {"name": "GW150914", "type": "BBH",  "duration": 32},
    {"name": "GW151226", "type": "BBH",  "duration": 32},
    {"name": "GW190521", "type": "BBH",  "duration": 32},
    {"name": "GW190814", "type": "NSBH", "duration": 32},
    {"name": "GW151012", "type": "BBH",  "duration": 32},
    {"name": "GW170104", "type": "BBH",  "duration": 32},
    {"name": "GW170608", "type": "BBH",  "duration": 32},
]

QUIET_SEGMENTS = [
    {"name": "O3a_q01_L1", "gps_start": 1238166018, "detector": "L1", "duration": 4096},
    {"name": "O3a_q01_H1", "gps_start": 1238166018, "detector": "H1", "duration": 4096},
    {"name": "O3a_q12_L1", "gps_start": 1242578176, "detector": "L1", "duration": 4096},
    {"name": "O3a_q12_H1", "gps_start": 1242578176, "detector": "H1", "duration": 4096},
]

# ─────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────

def fetch_event_strain(event_name, detector='L1', duration=32):
    try:
        import requests, h5py, time
        from gwosc.locate import get_event_urls
        time.sleep(2.0)  # rate limit pause
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
        return strain
    except Exception as e:
        print(f"    [WARN] {detector}: {e}")
        return None

def fetch_bulk_strain(gps_start, duration, detector='L1'):
    try:
        import requests, h5py, time
        from gwosc.locate import get_urls
        time.sleep(2.0)  # rate limit pause
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
        print(f"    [WARN] {detector}: {e}")
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
    if len(tmpl) < n:
        tmpl = np.pad(tmpl, (0, n-len(tmpl)))
    else:
        tmpl = tmpl[:n]
    return tmpl

def get_snr_timeseries(strain):
    """Връща SNR time series и peak info."""
    cal  = int(5.0 * FS)
    w    = whiten(strain)
    w   /= (np.std(w[:cal]) + 1e-12)
    n    = len(w)

    best_snr_t = np.zeros(n)
    for cfg in TEMPLATES:
        tmpl  = make_template(cfg, n)
        corr  = signal.correlate(w, tmpl, mode='same')
        med   = np.median(corr)
        mad   = 1.4826 * np.median(np.abs(corr - med)) + 1e-20
        snr_t = np.abs(corr - med) / mad
        best_snr_t = np.maximum(best_snr_t, snr_t)

    return best_snr_t

def get_triggers(snr_t, threshold, min_gap_s=0.1):
    """Намира trigger времена."""
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

def is_glitch(strain, t_trigger, window=1.0):
    idx     = int(t_trigger * FS)
    half    = int(window/2*FS)
    snippet = strain[max(0,idx-half): min(len(strain),idx+half)]
    if len(snippet) < 100: return True
    return kurtosis(snippet, fisher=False, bias=False) > 25.0

def frequency_line_veto(strain, t_trigger, window=1.0, line_freqs=[60, 120, 180]):
    """
    Veto за instrumental lines (60Hz power line и хармоници).
    Ако доминантната честота е близо до известна line — reject.
    """
    idx     = int(t_trigger * FS)
    half    = int(window/2*FS)
    snippet = strain[max(0,idx-half): min(len(strain),idx+half)]
    if len(snippet) < 100: return False

    freqs = np.fft.rfftfreq(len(snippet), 1/FS)
    power = np.abs(np.fft.rfft(snippet))**2
    mask  = (freqs > 20) & (freqs < 500)
    if not np.any(mask): return False

    f_peak = float(freqs[mask][np.argmax(power[mask])])
    for lf in line_freqs:
        if abs(f_peak - lf) < 2.0:  # 2Hz tolerance
            return True
    return False

def psd_ratio_veto(strain, t_trigger, cal_s=5.0, ratio_threshold=5.0):
    """
    Veto за scattered light — проверява дали ниските честоти (20-50Hz)
    доминират над сигналния диапазон (50-300Hz).
    Scattered light има excess power при ниски честоти.
    """
    idx     = int(t_trigger * FS)
    half    = int(1.0*FS)
    snippet = strain[max(0,idx-half): min(len(strain),idx+half)]
    if len(snippet) < FS//2: return False

    freqs    = np.fft.rfftfreq(len(snippet), 1/FS)
    power    = np.abs(np.fft.rfft(snippet))**2

    low_mask  = (freqs >= 20) & (freqs < 50)
    high_mask = (freqs >= 50) & (freqs < 300)

    if not np.any(low_mask) or not np.any(high_mask): return False

    low_power  = np.mean(power[low_mask])  + 1e-30
    high_power = np.mean(power[high_mask]) + 1e-30
    ratio      = low_power / high_power

    return ratio > ratio_threshold

def combined_veto(strain, t_trigger):
    """Комбинира всички veto методи."""
    if is_glitch(strain, t_trigger):
        return True, 'kurtosis'
    if frequency_line_veto(strain, t_trigger):
        return True, 'freq_line'
    if psd_ratio_veto(strain, t_trigger):
        return True, 'psd_ratio'
    return False, None

# ─────────────────────────────────────────────────────────────
# 1. Multi-detector coincidence
# ─────────────────────────────────────────────────────────────

def run_coincidence(events):
    """
    За всеки GW евент изтегля L1 и H1 данни.
    Проверява дали ORAC-NT trigger-ва и в двата детектора
    в рамките на COINCIDENCE_WINDOW (10ms).
    """
    print("\n📡 PHASE C-1: MULTI-DETECTOR COINCIDENCE")
    print(f"   Coincidence window: {COINCIDENCE_WINDOW*1000:.0f}ms (LIGO H1-L1 light travel)")
    print("-" * 62)

    results = []
    for ev in events:
        print(f"\n  [{ev['name']}] {ev['type']}")

        l1 = fetch_event_strain(ev['name'], 'L1', ev['duration'])
        h1 = fetch_event_strain(ev['name'], 'H1', ev['duration'])

        if l1 is None or h1 is None:
            print(f"    ⚠️  Missing detector data")
            results.append({**ev, "coincidence": False, "l1_snr": 0, "h1_snr": 0})
            continue

        center = len(l1)/FS/2.0

        # SNR time series за двата детектора
        snr_l1 = get_snr_timeseries(l1)
        snr_h1 = get_snr_timeseries(h1)

        # Triggers
        trg_l1 = get_triggers(snr_l1, CALIBRATED_T)
        trg_h1 = get_triggers(snr_h1, CALIBRATED_T)

        # Triggers в ±5s прозорец около евента
        trg_l1_w = [(t,s) for t,s in trg_l1 if abs(t-center) < 5.0]
        trg_h1_w = [(t,s) for t,s in trg_h1 if abs(t-center) < 5.0]

        # Coincidence check
        coincidence = False
        best_l1_snr = max((s for _,s in trg_l1_w), default=0.0)
        best_h1_snr = max((s for _,s in trg_h1_w), default=0.0)

        for t_l1, s_l1 in trg_l1_w:
            for t_h1, s_h1 in trg_h1_w:
                if abs(t_l1 - t_h1) <= COINCIDENCE_WINDOW:
                    if not is_glitch(l1, t_l1) and not is_glitch(h1, t_h1):
                        coincidence = True
                        break

        status = "✅ COINCIDENCE" if coincidence else "❌ NO COINCIDENCE"
        print(f"    {status} | L1 SNR={best_l1_snr:.1f} | H1 SNR={best_h1_snr:.1f}")
        print(f"    L1 triggers in window: {len(trg_l1_w)} | H1 triggers: {len(trg_h1_w)}")

        results.append({
            **ev,
            "coincidence": coincidence,
            "l1_snr": best_l1_snr,
            "h1_snr": best_h1_snr
        })

    confirmed = sum(1 for r in results if r['coincidence'])
    print(f"\n  Coincidence Rate: {confirmed}/{len(results)}")
    return results

# ─────────────────────────────────────────────────────────────
# 2. Glitch rejection benchmark
# ─────────────────────────────────────────────────────────────

def make_glitch(glitch_type, duration=2.0):
    """Генерира синтетичен glitch."""
    n  = int(duration * FS)
    t  = np.linspace(0, duration, n)

    if glitch_type == 'blip':
        # Blip: кратък широколентов burst ~10ms
        g = 20.0 * np.exp(-((t - duration/2)**2) / (2 * 0.01**2))
        g *= np.sin(2*np.pi*150*t)

    elif glitch_type == 'scattered_light':
        # Scattered light: повтарящи се дъги при ниски честоти
        g = 5.0 * np.sin(2*np.pi*30*t) * (1 + 0.5*np.sin(2*np.pi*3*t))
        g *= np.exp(-t/0.5)

    elif glitch_type == 'line':
        # Instrumental line: монохроматичен при фиксирана честота
        g = 8.0 * np.sin(2*np.pi*60*t)  # 60Hz power line

    elif glitch_type == 'burst':
        # Loud burst: широколентов с висока амплитуда
        g = 15.0 * np.random.randn(n) * np.exp(-((t-duration/2)**2)/(2*0.05**2))

    else:
        g = np.zeros(n)

    return g

def run_glitch_rejection(quiet_strains):
    """
    Инжектира различни типове glitch-ове в реален шум
    и измерва rejection rate на kurtosis veto.
    """
    print("\n\n📡 PHASE C-2: GLITCH REJECTION BENCHMARK")
    print("-" * 62)

    glitch_types  = ['blip', 'scattered_light', 'line', 'burst']
    n_trials      = 30
    results       = {}

    for gt in glitch_types:
        rejected = 0
        triggered_first = 0

        for trial in range(n_trials):
            # Случаен 4s прозорец от тихи данни
            qs        = quiet_strains[trial % len(quiet_strains)]
            start_idx = np.random.randint(0, len(qs) - 4*FS)
            noise     = qs[start_idx: start_idx + 4*FS].copy()

            # Инжектирай glitch в центъра
            g     = make_glitch(gt, duration=4.0)
            strain = noise + g[:len(noise)]

            t_center = 2.0

            # Комбиниран veto
            vetoed, veto_reason = combined_veto(strain, t_center)
            if vetoed:
                rejected += 1

            # Проверка дали trigger-ва въобще
            snr_t  = get_snr_timeseries(strain)
            trgs   = get_triggers(snr_t, CALIBRATED_T)
            near   = [t for t,_ in trgs if abs(t - t_center) < 1.0]
            if near:
                triggered_first += 1

        rejection_rate = rejected / n_trials * 100
        trigger_rate   = triggered_first / n_trials * 100
        results[gt]    = {"rejection": rejection_rate, "trigger": trigger_rate}

        status = "✅" if rejection_rate >= 70 else "⚠️ " if rejection_rate >= 40 else "❌"
        print(f"  {status} [{gt:<16}] Triggered: {trigger_rate:.0f}%  |  Vetoed: {rejection_rate:.0f}%")

    return results

# ─────────────────────────────────────────────────────────────
# 3. Latency metrics
# ─────────────────────────────────────────────────────────────

def run_latency_test(strain_sample):
    """
    Измерва латентност на всяка стъпка от pipeline-а.
    """
    print("\n\n📡 PHASE C-3: LATENCY METRICS")
    print("-" * 62)

    n_trials = 20
    latencies = {
        "whitening_ms":    [],
        "snr_compute_ms":  [],
        "trigger_find_ms": [],
        "total_ms":        [],
    }

    for _ in range(n_trials):
        t0 = time.perf_counter()
        w  = whiten(strain_sample)
        t1 = time.perf_counter()

        snr_t = get_snr_timeseries(strain_sample)
        t2 = time.perf_counter()

        trgs = get_triggers(snr_t, CALIBRATED_T)
        t3 = time.perf_counter()

        latencies["whitening_ms"].append((t1-t0)*1000)
        latencies["snr_compute_ms"].append((t2-t1)*1000)
        latencies["trigger_find_ms"].append((t3-t2)*1000)
        latencies["total_ms"].append((t3-t0)*1000)

    print(f"  {'Stage':<20} {'Mean (ms)':>10} {'Std (ms)':>10}")
    print(f"  {'-'*42}")
    for stage, vals in latencies.items():
        mean = np.mean(vals)
        std  = np.std(vals)
        print(f"  {stage:<20} {mean:>10.1f} {std:>10.1f}")

    print(f"\n  Note: Software latency on CPU.")
    print(f"  STM32F401 hardware latency: 535ns (documented)")

    return latencies

# ─────────────────────────────────────────────────────────────
# 4. ROC with coincidence
# ─────────────────────────────────────────────────────────────

def run_roc_coincidence(coincidence_results, quiet_l1, quiet_h1):
    """
    ROC curve с H1+L1 coincidence requirement.
    Signal = GW евенти с coincidence
    Noise  = случайни прозорци без coincidence
    """
    print("\n\n📡 PHASE C-4: ROC WITH COINCIDENCE")
    print("-" * 62)

    signal_snrs = []
    for r in coincidence_results:
        if r['coincidence']:
            # Combined SNR = sqrt(L1² + H1²)
            combined = np.sqrt(r['l1_snr']**2 + r['h1_snr']**2)
            signal_snrs.append(combined)

    noise_snrs = []
    min_len = min(len(quiet_l1), len(quiet_h1))
    for i in range(0, min_len - 32*FS, 32*FS):
        chunk_l1 = quiet_l1[i:i+32*FS]
        chunk_h1 = quiet_h1[i:i+32*FS]
        snr_l1   = float(np.max(get_snr_timeseries(chunk_l1)))
        snr_h1   = float(np.max(get_snr_timeseries(chunk_h1)))
        combined  = np.sqrt(snr_l1**2 + snr_h1**2)
        noise_snrs.append(combined)

    print(f"  Signal samples: {len(signal_snrs)}")
    print(f"  Noise samples:  {len(noise_snrs)}")

    if not signal_snrs or not noise_snrs:
        return [], [], [], 0.5

    all_snrs = sorted(set(signal_snrs + noise_snrs), reverse=True)
    tprs, fprs = [0.0], [0.0]

    for thr in all_snrs:
        tpr = sum(1 for s in signal_snrs if s >= thr) / len(signal_snrs)
        fpr = sum(1 for n in noise_snrs  if n >= thr) / len(noise_snrs)
        tprs.append(tpr); fprs.append(fpr)

    tprs.append(1.0); fprs.append(1.0)

    pairs  = sorted(zip(fprs, tprs))
    s_fprs = [p[0] for p in pairs]
    s_tprs = [p[1] for p in pairs]
    auc    = float(np.trapz(s_tprs, s_fprs))
    print(f"  AUC (coincidence) = {auc:.3f}")

    return all_snrs, tprs, fprs, auc

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run_phase_c():

    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("\n" + "=" * 62)
    print("  ORAC-NT PHASE C BENCHMARK")
    print(f"  {run_utc}")
    print("=" * 62)

    # ── Load quiet data (L1 + H1) ─────────────────────────
    print("\n📥 LOADING QUIET SEGMENTS (L1 + H1)")
    quiet_data = {}
    for seg in QUIET_SEGMENTS:
        print(f"\n  [{seg['name']}] GPS={seg['gps_start']}")
        s = fetch_bulk_strain(seg['gps_start'], seg['duration'], seg['detector'])
        if s is not None:
            quiet_data[seg['name']] = s
            print(f"    [OK] {len(s)/FS:.0f}s @ {FS} Hz")

    quiet_l1 = quiet_data.get('O3a_q01_L1', quiet_data.get('O3a_q12_L1'))
    quiet_h1 = quiet_data.get('O3a_q01_H1', quiet_data.get('O3a_q12_H1'))
    quiet_strains = [v for v in quiet_data.values() if v is not None]

    # ── C1: Coincidence ───────────────────────────────────
    coinc_results = run_coincidence(GW_EVENTS_COINCIDENCE)

    # ── C2: Glitch rejection ──────────────────────────────
    if quiet_strains:
        glitch_results = run_glitch_rejection(quiet_strains)
    else:
        glitch_results = {}
        print("  [SKIP] No quiet data for glitch test")

    # ── C3: Latency ───────────────────────────────────────
    sample_strain = quiet_strains[0][:32*FS] if quiet_strains else np.random.randn(32*FS)
    latency_results = run_latency_test(sample_strain)

    # ── C4: ROC with coincidence ──────────────────────────
    if quiet_l1 is not None and quiet_h1 is not None:
        _, roc_tprs, roc_fprs, roc_auc = run_roc_coincidence(
            coinc_results, quiet_l1, quiet_h1
        )
    else:
        roc_tprs, roc_fprs, roc_auc = [0,1], [0,1], 0.5
        print("\n  [SKIP] H1 data unavailable for coincidence ROC")

    # ── Final summary ─────────────────────────────────────
    print("\n\n" + "=" * 62)
    print("  PHASE C RESULTS")
    print("=" * 62)

    coinc_count = sum(1 for r in coinc_results if r['coincidence'])
    print(f"\n  H1+L1 Coincidence : {coinc_count}/{len(coinc_results)}")

    if glitch_results:
        avg_rejection = np.mean([v['rejection'] for v in glitch_results.values()])
        print(f"  Glitch rejection  : {avg_rejection:.0f}% average")

    avg_latency = np.mean(latency_results['total_ms'])
    print(f"  Pipeline latency  : {avg_latency:.0f}ms (software)")
    print(f"  Hardware latency  : 535ns (STM32F401, documented)")
    print(f"  ROC AUC (coinc.)  : {roc_auc:.3f}")

    # ── Visualization ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0e1117')

    # Panel 1 — Coincidence results
    names    = [r['name'] for r in coinc_results]
    l1_snrs  = [r['l1_snr'] for r in coinc_results]
    h1_snrs  = [r['h1_snr'] for r in coinc_results]
    x        = np.arange(len(names))
    axes[0,0].bar(x-0.2, l1_snrs, 0.4, color='#00ffcc', alpha=0.85, label='L1 SNR')
    axes[0,0].bar(x+0.2, h1_snrs, 0.4, color='#4488ff', alpha=0.85, label='H1 SNR')
    axes[0,0].axhline(CALIBRATED_T, color='red', lw=1.5, ls='--', label=f'Threshold={CALIBRATED_T}')
    axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(names, rotation=45, color='white', fontsize=8)
    axes[0,0].set_title(f"H1+L1 Coincidence — {coinc_count}/{len(coinc_results)} confirmed",
                        color='white', fontsize=10)
    axes[0,0].set_ylabel("Peak SNR", color='white')
    axes[0,0].legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
    axes[0,0].set_facecolor('#0e1117'); axes[0,0].tick_params(colors='white')

    # Panel 2 — Glitch rejection
    if glitch_results:
        gt_names   = list(glitch_results.keys())
        rejections = [glitch_results[g]['rejection'] for g in gt_names]
        triggers   = [glitch_results[g]['trigger']   for g in gt_names]
        x2 = np.arange(len(gt_names))
        axes[0,1].bar(x2-0.2, triggers,   0.4, color='#ff4444', alpha=0.85, label='Triggered (%)')
        axes[0,1].bar(x2+0.2, rejections, 0.4, color='#00ffcc', alpha=0.85, label='Vetoed (%)')
        axes[0,1].set_xticks(x2); axes[0,1].set_xticklabels(gt_names, rotation=20, color='white', fontsize=9)
        axes[0,1].set_title("Glitch Rejection Rate (kurtosis veto)", color='white', fontsize=10)
        axes[0,1].set_ylabel("Rate (%)", color='white')
        axes[0,1].set_ylim(0, 110)
        axes[0,1].legend(facecolor='#0e1117', labelcolor='white', fontsize=8)
        axes[0,1].set_facecolor('#0e1117'); axes[0,1].tick_params(colors='white')

    # Panel 3 — Latency
    stages = ['whitening_ms', 'snr_compute_ms', 'trigger_find_ms']
    labels = ['Whitening', 'SNR Compute', 'Trigger Find']
    means  = [np.mean(latency_results[s]) for s in stages]
    stds   = [np.std(latency_results[s])  for s in stages]
    colors = ['#00ffcc', '#4488ff', '#ff9900']
    bars   = axes[1,0].bar(labels, means, color=colors, alpha=0.85, yerr=stds,
                           capsize=5, error_kw={'color': 'white'})
    axes[1,0].set_title("Pipeline Latency per Stage (Software, CPU)", color='white', fontsize=10)
    axes[1,0].set_ylabel("Latency (ms)", color='white')
    axes[1,0].set_facecolor('#0e1117'); axes[1,0].tick_params(colors='white')
    for bar, mean in zip(bars, means):
        axes[1,0].text(bar.get_x()+bar.get_width()/2, mean+5,
                       f'{mean:.0f}ms', ha='center', color='white', fontsize=9)
    axes[1,0].text(0.98, 0.95, 'STM32F401: 535ns',
                   transform=axes[1,0].transAxes, color='#ffaa00',
                   fontsize=10, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))

    # Panel 4 — ROC with coincidence
    if len(roc_fprs) > 2:
        axes[1,1].plot(roc_fprs, roc_tprs, color='#00d2ff', lw=2.5,
                       label=f'ORAC-NT H1+L1 (AUC={roc_auc:.3f})')
        axes[1,1].plot([0,1], [0,1], color='gray', ls='--', lw=1, alpha=0.5, label='Random')
        axes[1,1].set_xlim(-0.02, 1.02); axes[1,1].set_ylim(-0.02, 1.02)
    else:
        axes[1,1].text(0.5, 0.5, f'AUC = {roc_auc:.3f}\n(H1 data required)',
                       transform=axes[1,1].transAxes, color='white',
                       fontsize=12, ha='center', va='center')
    axes[1,1].set_title("ROC Curve with H1+L1 Coincidence", color='white', fontsize=10)
    axes[1,1].set_xlabel("False Positive Rate", color='white')
    axes[1,1].set_ylabel("True Positive Rate", color='white')
    axes[1,1].legend(facecolor='#0e1117', labelcolor='white', fontsize=9)
    axes[1,1].set_facecolor('#0e1117'); axes[1,1].tick_params(colors='white')

    plt.suptitle(
        f"ORAC-NT Phase C — Coincidence + Glitch Rejection + Latency\n"
        f"Dimitar Kretski  |  DOI: 10.5281/zenodo.20098932  |  {run_utc}",
        color='white', fontsize=11
    )
    plt.tight_layout()
    plt.savefig("orac_phase_c.png", dpi=150, facecolor='#0e1117', bbox_inches='tight')
    print(f"\n✅ orac_phase_c.png")
    print(f"   🕐 {run_utc}")

if __name__ == "__main__":
    run_phase_c()
