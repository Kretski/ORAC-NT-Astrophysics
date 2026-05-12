import os
import numpy as np
import matplotlib.pyplot as plt

from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.types import TimeSeries

# ============================================================
# CONFIG
# ============================================================

FS = 4096
DURATION = 32

BANDS = 4

TARGET_SNR = 12.0

SNR_THRESHOLD = 8.0
CHISQ_THRESHOLD = 2.5

# ============================================================
# NOISE
# ============================================================

def make_noise(seed):

    flen = FS * DURATION // 2 + 1

    psd = aLIGOZeroDetHighPower(
        flen,
        1.0 / DURATION,
        20
    )

    noise = noise_from_psd(
        FS * DURATION,
        1.0 / FS,
        psd,
        seed=seed
    )

    return noise, psd


# ============================================================
# PROPER CHI-SQUARED
# ============================================================

def allen_chisq_simple(
    template,
    data,
    psd,
    nbands=BANDS
):
    """
    Proper normalized chi-squared discriminator.
    """

    # --------------------------------------------------------
    # Full matched filter
    # --------------------------------------------------------

    full_snr = matched_filter(
        template,
        data,
        psd=psd,
        low_frequency_cutoff=20
    )

    full_peak = float(
        abs(full_snr).crop(4, 4).max()
    )

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    htilde = template.to_frequencyseries()

    freqs = np.arange(len(htilde)) * htilde.delta_f

    fmin = 20
    fmax = min(1024, freqs[-1])

    edges = np.linspace(
        fmin,
        fmax,
        nbands + 1
    )

    band_snrs = []

    # --------------------------------------------------------
    # Split into bands
    # --------------------------------------------------------

    for i in range(nbands):

        low = edges[i]
        high = edges[i + 1]

        mask = np.zeros(len(htilde))

        idx = np.where(
            (freqs >= low) &
            (freqs < high)
        )[0]

        mask[idx] = 1.0

        band_htilde = htilde * mask

        band_template = band_htilde.to_timeseries(
            delta_t=template.delta_t
        )

        try:

            band_snr_ts = matched_filter(
                band_template,
                data,
                psd=psd,
                low_frequency_cutoff=20
            )

            band_peak = float(
                abs(band_snr_ts).crop(4, 4).max()
            )

        except:

            band_peak = 0.0

        band_snrs.append(band_peak)

    band_snrs = np.array(band_snrs)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    total = np.sum(band_snrs)

    if total <= 1e-12:

        return 1e9, band_snrs

    band_frac = band_snrs / total

    expected = np.ones(nbands) / nbands

    # --------------------------------------------------------
    # Proper reduced chi²
    # --------------------------------------------------------

    chisq = nbands * np.sum(
        (band_frac - expected) ** 2
    )

    return chisq, band_snrs


# ============================================================
# REWEIGHTED SNR
# ============================================================

def reweighted_snr(snr, chisq):

    if chisq <= 1.0:
        return snr

    return snr / np.sqrt(chisq)


# ============================================================
# TEMPLATE PREP
# ============================================================

def prepare_template(noise):

    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=30,
        mass2=30,
        delta_t=1.0 / FS,
        f_lower=20
    )

    print(f" Raw waveform length: {len(hp)}")

    keep = FS * 4

    if len(hp) > keep:

        start = len(hp) - keep

        hp = hp[start:len(hp)]

    print(f" Cropped waveform length: {len(hp)}")

    hp.start_time = noise.start_time

    if len(hp) < len(noise):

        hp.resize(len(noise))

    return hp


# ============================================================
# SELF CALIBRATION
# ============================================================

def compute_scale_factor(template, psd):

    self_snr = matched_filter(
        template,
        template,
        psd=psd,
        low_frequency_cutoff=20
    )

    peak = float(
        abs(self_snr).crop(4, 4).max()
    )

    print(f" Self-template peak SNR: {peak:.3f}")

    scale = TARGET_SNR / peak

    print(f" Scale factor: {scale:.6f}")

    return scale


# ============================================================
# MAIN
# ============================================================

def run():

    os.makedirs("outputs", exist_ok=True)

    print("🚀 ORAC Phase E5 — FIXED Chi² Pipeline")
    print("=" * 70)

    # ========================================================
    # TEMPLATE
    # ========================================================

    noise0, psd0 = make_noise(0)

    template = prepare_template(noise0)

    scale = compute_scale_factor(
        template,
        psd0
    )

    # ========================================================
    # SIGNAL TESTS
    # ========================================================

    signal_events = []

    print("\nTesting REAL signals...")

    for i in range(100):

        noise, psd = make_noise(i + 1000)

        data = noise + template * scale

        snr_ts = matched_filter(
            template,
            data,
            psd=psd,
            low_frequency_cutoff=20
        )

        peak = float(
            abs(snr_ts).crop(4, 4).max()
        )

        chisq, bands = allen_chisq_simple(
            template,
            data,
            psd
        )

        rw_snr = reweighted_snr(
            peak,
            chisq
        )

        signal_events.append({
            'snr': peak,
            'chisq': chisq,
            'rw': rw_snr
        })

        if (i + 1) % 20 == 0:

            print(
                f" Progress {i+1}/100 | "
                f"SNR={peak:.2f} | "
                f"χ²={chisq:.2f} | "
                f"ρ̂={rw_snr:.2f}"
            )

    # ========================================================
    # NOISE TESTS
    # ========================================================

    print("\nTesting NOISE...")

    noise_events = []

    for i in range(200):

        noise, psd = make_noise(i + 5000)

        snr_ts = matched_filter(
            template,
            noise,
            psd=psd,
            low_frequency_cutoff=20
        )

        peak = float(
            abs(snr_ts).crop(4, 4).max()
        )

        chisq, _ = allen_chisq_simple(
            template,
            noise,
            psd
        )

        rw_snr = reweighted_snr(
            peak,
            chisq
        )

        noise_events.append({
            'snr': peak,
            'chisq': chisq,
            'rw': rw_snr
        })

    # ========================================================
    # GLITCH TESTS
    # ========================================================

    print("\nTesting GLITCHES...")

    glitch_events = []

    for i in range(100):

        noise, psd = make_noise(i + 10000)

        gdata = noise.numpy().copy()

        loc = np.random.randint(
            len(gdata)//4,
            3*len(gdata)//4
        )

        width = 30

        amp = 25 * np.std(gdata)

        gdata[loc:loc+width] += amp

        glitch_ts = TimeSeries(
            gdata,
            delta_t=noise.delta_t
        )

        glitch_ts.start_time = noise.start_time

        snr_ts = matched_filter(
            template,
            glitch_ts,
            psd=psd,
            low_frequency_cutoff=20
        )

        peak = float(
            abs(snr_ts).crop(4, 4).max()
        )

        chisq, _ = allen_chisq_simple(
            template,
            glitch_ts,
            psd
        )

        rw_snr = reweighted_snr(
            peak,
            chisq
        )

        glitch_events.append({
            'snr': peak,
            'chisq': chisq,
            'rw': rw_snr
        })

    # ========================================================
    # PIPELINE RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)

    sig_triggers = [
        e for e in signal_events
        if e['rw'] > SNR_THRESHOLD
    ]

    noi_triggers = [
        e for e in noise_events
        if e['rw'] > SNR_THRESHOLD
    ]

    gli_triggers = [
        e for e in glitch_events
        if e['rw'] > SNR_THRESHOLD
    ]

    sig_kept = [
        e for e in sig_triggers
        if e['chisq'] < CHISQ_THRESHOLD
    ]

    gli_kept = [
        e for e in gli_triggers
        if e['chisq'] < CHISQ_THRESHOLD
    ]

    print()

    print(f"Signals above threshold: {len(sig_triggers)}/100")
    print(f"Noise above threshold: {len(noi_triggers)}/200")
    print(f"Glitches above threshold: {len(gli_triggers)}/100")

    print()

    print(
        f"Signals kept after χ² veto: "
        f"{len(sig_kept)}/{max(len(sig_triggers),1)}"
    )

    print(
        f"Glitches kept after χ² veto: "
        f"{len(gli_kept)}/{max(len(gli_triggers),1)}"
    )

    # ========================================================
    # PLOTS
    # ========================================================

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    # --------------------------------------------------------

    ax[0].scatter(
        [e['snr'] for e in signal_events],
        [e['chisq'] for e in signal_events],
        alpha=0.6,
        label='Signals'
    )

    ax[0].scatter(
        [e['snr'] for e in glitch_events],
        [e['chisq'] for e in glitch_events],
        alpha=0.6,
        label='Glitches'
    )

    ax[0].axhline(
        CHISQ_THRESHOLD,
        color='red',
        linestyle='--'
    )

    ax[0].set_xlabel("Raw SNR")
    ax[0].set_ylabel("Chi²")
    ax[0].set_title("Chi² Discriminator")
    ax[0].grid(True)
    ax[0].legend()

    # --------------------------------------------------------

    ax[1].hist(
        [e['rw'] for e in signal_events],
        bins=20,
        alpha=0.7,
        label='Signals',
        density=True
    )

    ax[1].hist(
        [e['rw'] for e in glitch_events],
        bins=20,
        alpha=0.7,
        label='Glitches',
        density=True
    )

    ax[1].axvline(
        SNR_THRESHOLD,
        color='red',
        linestyle='--'
    )

    ax[1].set_xlabel("Reweighted SNR")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Reweighted Ranking")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()

    save_path = "outputs/e5_chisq_reweighted_FIXED.png"

    plt.savefig(
        save_path,
        dpi=150
    )

    print(f"\n✅ Saved: {save_path}")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    run()