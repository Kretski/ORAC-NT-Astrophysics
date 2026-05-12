import os
import numpy as np
import matplotlib.pyplot as plt

from pycbc.filter import matched_filter
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.waveform import get_td_waveform
from pycbc.types import TimeSeries

# ============================================================
# CONFIG
# ============================================================

FS = 4096
DURATION = 32
TARGET_SNR = 12.0

# ============================================================
# BUILD TEMPLATE
# ============================================================

def build_template():

    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=23,
        mass2=2.6,
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

    return hp

# ============================================================
# MAIN
# ============================================================

def main():

    print("🔍 REAL SELF-CALIBRATED SNR TEST")

    # ========================================================
    # PSD
    # ========================================================

    flen = FS * DURATION // 2 + 1

    psd = aLIGOZeroDetHighPower(
        flen,
        1.0 / DURATION,
        20
    )

    # ========================================================
    # Noise
    # ========================================================

    noise = noise_from_psd(
        FS * DURATION,
        1.0 / FS,
        psd,
        seed=42
    )

    print(f" Noise length: {len(noise)}")

    # ========================================================
    # Template
    # ========================================================

    hp = build_template()

    # ========================================================
    # Pad template
    # ========================================================

    template_array = np.zeros(len(noise))

    center = len(noise) // 2

    start = center - len(hp) // 2

    end = start + len(hp)

    template_array[start:end] = hp.numpy()

    template = TimeSeries(
        template_array,
        delta_t=1.0 / FS
    )

    print(f" Template centered at sample: {center}")

    # ========================================================
    # SELF SNR CALIBRATION
    # ========================================================

    pure = matched_filter(
        template,
        template,
        psd=psd,
        low_frequency_cutoff=20
    )

    pure = pure.crop(4, 4)

    pure_peak = float(abs(pure).max())

    print(f" Self template SNR: {pure_peak:.4f}")

    # ========================================================
    # SCALE FACTOR
    # ========================================================

    scale = TARGET_SNR / pure_peak

    print(f" Scale factor: {scale:.6f}")

    injection = template * scale

    # ========================================================
    # Inject into noise
    # ========================================================

    data = noise + injection

    print(" ✅ Injection complete")

    # ========================================================
    # Recovery matched filter
    # ========================================================

    snr = matched_filter(
        template,
        data,
        psd=psd,
        low_frequency_cutoff=20
    )

    snr = snr.crop(4, 4)

    snr_abs = abs(snr)

    peak = float(snr_abs.max())

    peak_idx = int(snr_abs.abs_arg_max())

    # ========================================================
    # Results
    # ========================================================

    error = abs(peak - TARGET_SNR) / TARGET_SNR * 100

    print("\n📊 RESULTS")
    print(f" Target SNR  : {TARGET_SNR:.2f}")
    print(f" Measured SNR: {peak:.2f}")
    print(f" Error       : {error:.2f}%")

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(14,5))

    # --------------------------------------------------------

    plt.subplot(1,2,1)

    plt.plot(
        snr_abs.numpy(),
        linewidth=1
    )

    plt.axhline(
        8,
        color='red',
        linestyle='--',
        label='Threshold'
    )

    plt.axvline(
        peak_idx,
        color='green',
        linestyle=':',
        label=f'Peak={peak:.2f}'
    )

    plt.title("Matched Filter Output")
    plt.xlabel("Sample")
    plt.ylabel("|SNR|")

    plt.grid(True)
    plt.legend()

    # --------------------------------------------------------

    plt.subplot(1,2,2)

    plt.hist(
        snr_abs.numpy(),
        bins=120,
        alpha=0.75
    )

    plt.axvline(
        peak,
        color='red',
        linestyle='--',
        label=f'Peak={peak:.2f}'
    )

    plt.title("SNR Distribution")
    plt.xlabel("SNR")
    plt.ylabel("Count")

    plt.grid(True)
    plt.legend()

    # --------------------------------------------------------

    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)

    out = "outputs/e1_selfcalibrated.png"

    plt.savefig(out, dpi=150)

    print(f"\n✅ Saved: {out}")

    # ========================================================

    print("\n================================================")

    if error < 5:
        print("✅ CALIBRATION SUCCESS")
    else:
        print("⚠️ CALIBRATION OFFSET")

    print("================================================")

# ============================================================

if __name__ == "__main__":
    main()