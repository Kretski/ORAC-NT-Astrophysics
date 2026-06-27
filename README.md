# ORAC-NT — Gravitational-Wave Candidate Morphology Pipeline

A real-time pipeline that ingests gravitational-wave alerts via NASA GCN
Kafka and applies morphological analysis to candidate events.

**Important scope:** ORAC-NT is a *post-detection morphological classifier
and ranker*, not a standalone detector. It characterises and ranks candidate
events by their time-frequency morphology; it does not replace the matched-
filter detection pipelines (PyCBC, GstLAL) that produce the candidates.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20129975.svg)](https://doi.org/10.5281/zenodo.20129975)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## What it does, honestly

ORAC-NT performs:
- Real-time ingestion of GCN/IGWN alerts (Kafka listener, multi-topic)
- Bandpass + PSD whitening of strain segments
- Morphological feature extraction (matched-filter SNR, chi-squared
  consistency, kurtosis veto)
- BBH-vs-glitch morphological classification

### Validated capability

On real Gravity Spy glitch data, the morphological classifier separates
BBH-like signals from common glitch classes with **AUC ≈ 0.876**. ⚠️ [confirm this is your real Gravity Spy number]

### Honest limitation (important)

Against detector noise at realistic single-detector SNR (8–20), the
morphological features do **not** function as a signal-vs-noise
discriminator (AUC ≈ 0.475 — near chance). ORAC-NT ranks and characterises
candidates that a detection pipeline has already flagged; it is not itself
a detection statistic. Reported high separation on canonical event sets
reflects signal-*type* separation at high catalog SNR, not signal-vs-noise
detection.

---

## What the benchmark numbers mean (and don't)

The pipeline correctly flags canonical events when they are present in the
input. This demonstrates the morphology/ranking stage works on real signals
— it does **not** establish standalone detection sensitivity or a
calibrated false-alarm rate.

- No false alarms were observed in 7.96h of quiet O3 data. ⚠️ This is a
  limited-duration observation, NOT an IFAR or per-year FAR — 8 hours
  cannot establish long-timescale false-alarm rates.
- AUC figures refer to morphological separation on labeled sets, not
  signal-vs-noise detection.

[Махнати: "IFAR > 1,000,000 yr", "FAR = 0/yr" — статистически неустановими от наличните данни]

---

## Pipeline architecture

[остава както е — техническата диаграма е реална и полезна]

---

## Hardware note

The core scalar computation has been benchmarked on STM32F401 at ~535 ns
per evaluation. This is the latency of the arithmetic, not an end-to-end
detection latency, and is relevant only to the embedded-deployment variant.

---

## Installation / Usage

[остава както е — това е реално и полезно]

---

## Status & roadmap

Active research pipeline. The detection-sensitivity and false-alarm
characterisation are limited by available background data; current results
should be read as a working morphology/ranking stage, not a validated
standalone search.

---

## License

MIT — free for research use with attribution.
Data: GWOSC (LIGO Open Science Center), CC BY 4.0.

---

