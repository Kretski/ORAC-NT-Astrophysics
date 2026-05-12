import os
import numpy as np
import matplotlib.pyplot as plt

from pycbc.waveform import get_td_waveform
from pycbc.filter import matched_filter, sigma
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.noise import noise_from_psd
from pycbc.types import TimeSeries

# ============================================================
# CONFIG
# ============================================================

FS = 4096
DURATION = 32

ITERATIONS = 500

LOW_FREQ = 20

DETECTION_THRESHOLD = 8.0

# ============================================================
# PSD + NOISE
# ============================================================

def make_noise(seed):

    flen = FS * DURATION // 2 + 1

    psd = aLIGOZeroDetHighPower(
        flen,
        1.0 / DURATION,
        LOW_FREQ
    )

    noise = noise_from_psd(
        FS * DURATION,
        1.0 / FS,
        psd,
        seed=seed
    )

    return noise, psd


# ============================================================
# SHORT TEMPLATE
# ============================================================

def make_template():

    hp, _ = get_td_waveform(
        approximant="IMRPhenomD",
        mass1=20,
        mass2=20,
        delta_t=1.0 / FS,
        f_lower=LOW_FREQ
    )

    print(f" Raw waveform length: {len(hp)}")

    # --------------------------------------------------------
    # Keep only last 4 seconds
    # --------------------------------------------------------

    max_len = FS * 4

    if len(hp) > max_len:

        start = len(hp) - max_len

        hp = hp[start:len(hp)]

    print(f" Final waveform length: {len(hp)}")

    return hp


# ============================================================
# PAD TEMPLATE TO FULL LENGTH
# IMPORTANT:
# makes delta_f match PSD
# ============================================================

def pad_template(signal, total_len):

    arr = np.zeros(total_len)

    arr[:len(signal)] = signal.numpy()

    ts = TimeSeries(
        arr,
        delta_t=1.0 / FS
    )

    return ts


# ============================================================
# CENTER SIGNAL
# ============================================================

def center_signal(signal, total_len):

    arr = np.zeros(total_len)

    center = total_len // 2

    start = center - len(signal) // 2

    end = start + len(signal)

    arr[start:end] = signal.numpy()

    ts = TimeSeries(
        arr,
        delta_t=1.0 / FS
    )

    return ts


# ============================================================
# MAIN
# ============================================================

def run():

    os.makedirs("outputs", exist_ok=True)

    print("🚀 ORAC Phase E3 — REAL Injection Campaign")

    snr_levels = np.arange(2, 26, 2)

    efficiencies = []

    # ========================================================
    # CREATE SHORT TEMPLATE ONCE
    # ========================================================

    hp_short = make_template()

    # ========================================================
    # LOOP OVER TARGET SNR
    # ========================================================

    for target_snr in snr_levels:

        detected = 0

        print("\n================================================")
        print(f" TARGET SNR = {target_snr}")
        print("================================================")

        for i in range(ITERATIONS):

            # ------------------------------------------------
            # Noise + PSD
            # ------------------------------------------------

            noise, psd = make_noise(i)

            # ------------------------------------------------
            # Pad template to full length
            # IMPORTANT:
            # delta_f now matches PSD
            # ------------------------------------------------

            hp = pad_template(
                hp_short,
                len(noise)
            )

            # ------------------------------------------------
            # Compute sigma
            # ------------------------------------------------

            s = sigma(
                hp,
                psd=psd,
                low_frequency_cutoff=LOW_FREQ
            )

            scale_factor = target_snr / s

            print(f"\n Iteration {i+1}/{ITERATIONS}")
            print(f" Sigma        : {s:.6f}")
            print(f" Scale factor : {scale_factor:.8f}")

            # ------------------------------------------------
            # Scale waveform
            # ------------------------------------------------

            hp_scaled = hp_short * scale_factor

            # ------------------------------------------------
            # Inject centered signal
            # ------------------------------------------------

            injection = center_signal(
                hp_scaled,
                len(noise)
            )

            data = noise + injection

            # ------------------------------------------------
            # Pad scaled template
            # ------------------------------------------------

            hp_scaled_full = pad_template(
                hp_scaled,
                len(noise)
            )

            # ------------------------------------------------
            # Matched filter
            # ------------------------------------------------

            snr = matched_filter(
                hp_scaled_full,
                data,
                psd=psd,
                low_frequency_cutoff=LOW_FREQ
            )

            # ------------------------------------------------
            # Remove FFT edge corruption
            # ------------------------------------------------

            snr = snr.crop(4, 4)

            snr_abs = abs(snr)

            peak = float(snr_abs.max())

            peak_idx = snr_abs.abs_arg_max()

            print(f" Peak SNR     : {peak:.3f}")
            print(f" Peak sample  : {peak_idx}")

            # ------------------------------------------------
            # Detection threshold
            # ------------------------------------------------

            if peak > DETECTION_THRESHOLD:

                detected += 1

        # ====================================================
        # Detection efficiency
        # ====================================================

        efficiency = detected / ITERATIONS

        efficiencies.append(efficiency)

        print("\n------------------------------------------------")
        print(f" Detection efficiency: {efficiency:.3f}")
        print("------------------------------------------------")

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n================================================")
    print(" FINAL RESULTS")
    print("================================================")

    for s, e in zip(snr_levels, efficiencies):

        print(f" Injected SNR {s:2d} -> efficiency {e:.3f}")

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        snr_levels,
        efficiencies,
        marker='o',
        linewidth=2
    )

    plt.axhline(
        0.5,
        linestyle='--',
        label='50% Detection'
    )

    plt.axvline(
        DETECTION_THRESHOLD,
        linestyle=':',
        label='SNR 8 Threshold'
    )

    plt.xlabel("Injected SNR")

    plt.ylabel("Detection Efficiency")

    plt.title("ORAC Phase E3 — Sensitivity Curve")

    plt.grid(True, alpha=0.3)

    plt.ylim(-0.05, 1.05)

    plt.xlim(min(snr_levels)-1, max(snr_levels)+1)

    plt.legend()

    plt.tight_layout()

    out = "outputs/e3_injections_fixed.png"

    plt.savefig(out, dpi=150)

    print(f"\n✅ Saved: {out}")


# ============================================================

if __name__ == "__main__":

    run()