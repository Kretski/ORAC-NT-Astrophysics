# ORAC-NT: gravitational-wave candidate morphology (research code)

Research code for ingesting gravitational-wave alerts (NASA GCN Kafka) and
computing time-frequency morphology features for candidate events.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20129975.svg)](https://doi.org/10.5281/zenodo.20129975)

## Scope

ORAC-NT is intended as a **post-detection morphology and ranking stage**.
It is **not a detector** and does not replace matched-filter search pipelines
(PyCBC, GstLAL) that produce the candidates. No claim of standalone detection
sensitivity or of a calibrated false-alarm rate is made.

## What the code does

- GCN/IGWN alert listener (Kafka) - `orac_live.py`
- Strain bandpass and PSD whitening, morphological features (matched-filter
  SNR, chi-squared consistency, kurtosis veto) - `orac_nt_v10.py`, `orac_veto_layer.py`
- Analysis scripts by phase (`orac_phase_b` ... `orac_phase_e7`), time-slide
  and false-alarm experiments (`orac_time_slides.py`, `orac_far_benchmark_v4.py`)
- Reproducibility notebook - `ORAC_NT_Reproducibility.ipynb`

## Status of the numerical results

No performance number in this repository should currently be treated as a
validated result.

| Item | Status |
|---|---|
| Classifier AUC on real glitch data (Gravity Spy) | Not reproducible from this repository. Earlier drafts quoted a figure that no script or result file here produces; it has been removed. |
| Signal-vs-noise AUC at realistic SNR | Earlier drafts described it as near chance. No result file in this repository supports a specific value; treat the claim as "not demonstrated". |
| `orac_phase_e4_roc.py` | Draws scores from two synthetic Gaussians. Its AUC illustrates the ROC code only and says nothing about the method. |
| `results_time_slides/far_results.json` | Run on **simulated** O3-like noise, 3600 s, 20 time slides, 0 coincidences. This is an upper-bound-style check on simulated data, not a false-alarm rate on real detector data. A zero count does not establish "FAR = 0 per year". |
| "No false alarms in ~8 h of quiet O3 data" | Earlier drafts quoted this; the corresponding result file is not in this repository. Removed until it can be reproduced here. |
| STM32F401 ~535 ns per evaluation | Latency of one scalar computation only; not end-to-end detection latency. Source not included here. |

Reproducing any figure requires running the corresponding script on real
GWOSC data and committing its output file. Until then, the code is best read
as a working prototype of a feature-extraction and ranking stage.

## Installation

```
pip install -r requirements.txt
```
Data: GWOSC (LIGO Open Science Center), CC BY 4.0. Scripts assume network
access to GWOSC and, for `orac_live.py`, GCN credentials supplied by the user
through their own environment (never commit credentials).

## Citing

DOI: 10.5281/zenodo.20129975 (see also 10.5281/zenodo.19553825 for the earlier
record referenced in `LICENSE`).

## License

CC BY-NC 4.0, as stated in `LICENSE`: free for non-commercial research and
teaching with attribution; commercial use requires separate permission.
(If you intend MIT, replace `LICENSE` accordingly. The README and LICENSE
must say the same thing.)

Author: Dimitar Kretski, Varna, Bulgaria. ORCID 0000-0001-5108-2243.
