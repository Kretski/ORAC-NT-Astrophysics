import os
import numpy as np
import matplotlib.pyplot as plt

from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter
from pycbc.types import TimeSeries

# ============================================================
# CONFIG
# ============================================================

FS = 4096
DURATION = 32
N_SLIDES = 200

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
# TEMPLATE
# ============================================================

def build_template(noise_len):

    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=30,
        mass2=30,
        delta_t=1.0 / FS,
        f_lower=20
    )

    print(f" Raw waveform length: {len(hp)}")

    # --------------------------------------------------------
    # Safe crop
    # --------------------------------------------------------

    keep = FS * 4

    if len(hp) > keep:

        start_idx = len(hp) - keep

        hp = hp[start_idx:len(hp)]

    print(f" Cropped waveform length: {len(hp)}")

    # --------------------------------------------------------
    # Centered padding
    # --------------------------------------------------------

    template_array = np.zeros(noise_len)

    center = noise_len // 2

    start = center - len(hp) // 2

    end = start + len(hp)

    template_array[start:end] = hp.numpy()

    template = TimeSeries(
        template_array,
        delta_t=1.0 / FS
    )

    print(f" Template centered at sample: {center}")

    return template

# ============================================================
# MAIN
# ============================================================

def run():

    os.makedirs("outputs", exist_ok=True)

    print("🚀 ORAC Phase E2 — Time Slide FAR Analysis")

    # ========================================================
    # Initial noise for template sizing
    # ========================================================

    noise_ref, psd_ref = make_noise(0)

    # ========================================================
    # Build stable template
    # ========================================================

    template = build_template(len(noise_ref))

    # ========================================================
    # FAR background
    # ========================================================

    max_snrs = []

    for i in range(N_SLIDES):

        noise, psd = make_noise(i + 1)

        # ----------------------------------------------------
        # Random time slide
        # ----------------------------------------------------

        shift = np.random.randint(
            FS,
            FS * 10
        )

        rolled = np.roll(
            noise.numpy(),
            shift
        )

        rolled_ts = TimeSeries(
            rolled,
            delta_t=noise.delta_t
        )

        rolled_ts.start_time = noise.start_time

        # ----------------------------------------------------
        # Matched filter
        # ----------------------------------------------------

        snr = matched_filter(
            template,
            rolled_ts,
            psd=psd,
            low_frequency_cutoff=20
        )

        # ----------------------------------------------------
        # Remove FFT edge corruption
        # ----------------------------------------------------

        snr = snr.crop(4, 4)

        peak = float(abs(snr).max())

        max_snrs.append(peak)

        # ----------------------------------------------------

        if (i + 1) % 20 == 0:

            print(
                f" Progress: {i+1}/{N_SLIDES}"
                f" | Peak SNR: {peak:.2f}"
            )

    # ========================================================
    # Statistics
    # ========================================================

    max_snrs = np.array(max_snrs)

    mean_snr = np.mean(max_snrs)

    std_snr = np.std(max_snrs)

    threshold_5sigma = mean_snr + 5 * std_snr

    print("\n================================================")
    print(" BACKGROUND STATISTICS")
    print("================================================")

    print(f" Mean SNR     : {mean_snr:.3f}")
    print(f" Std SNR      : {std_snr:.3f}")
    print(f" 5σ Threshold : {threshold_5sigma:.3f}")

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(figsize=(11, 6))

    plt.hist(
        max_snrs,
        bins=35,
        alpha=0.75,
        edgecolor='black'
    )

    plt.axvline(
        mean_snr,
        linestyle='--',
        linewidth=2,
        label=f"Mean = {mean_snr:.2f}"
    )

    plt.axvline(
        threshold_5sigma,
        linestyle=':',
        linewidth=2,
        label=f"5σ = {threshold_5sigma:.2f}"
    )

    plt.xlabel("Maximum Noise SNR")
    plt.ylabel("Count")

    plt.title("ORAC Phase E2 — Time Slide Background")

    plt.grid(True, alpha=0.3)

    plt.legend()

    plt.tight_layout()

    out = "outputs/e2_timeslides.png"

    plt.savefig(out, dpi=150)

    print(f"\n✅ Saved: {out}")

# ============================================================

if __name__ == "__main__":
    run()