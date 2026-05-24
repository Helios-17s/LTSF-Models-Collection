# LTSF-Models-Collection

**Spatio-Temporal Flow Forecasting Across Physical Scales: A Multi-Scale Benchmark**

This repository contains all model implementations, training pipelines, evaluation
scripts, and figure-generation notebooks associated with the paper:

> *Spatio-Temporal Flow Forecasting Across Physical Scales: A Multi-Scale Benchmark
> from Epidemic Diffusion to National Mobility*  
> Submitted to *Nature Computational Science* (under review)

---

## Models Implemented

| Model | Architecture Family | Reference |
|---|---|---|
| STGCN | Graph Neural Network | Yu et al., IJCAI 2018 |
| DLinear | Linear Decomposition | Zeng et al., AAAI 2023 |
| SegRNN | Recurrent Network | Lin et al., arXiv 2023 |
| TimesNet | 2D Convolutional | Wu et al., ICLR 2023 |
| PatchTST | Transformer | Nie et al., ICLR 2023 |
| Informer | Transformer | Zhou et al., AAAI 2021 |
| Autoformer | Transformer | Wu et al., NeurIPS 2021 |
| Reformer | Transformer | Kitaev et al., ICLR 2020 |
| Transformer | Transformer | Vaswani et al., NeurIPS 2017 |

---

## Datasets

Five spatio-temporal flow datasets spanning four physical scales:

| Dataset | Scale | Nodes | Time Steps | Flow Type |
|---|---|---|---|---|
| Dengue citywide | Local | 119 | 468 (weekly) | Bus/transit |
| Korea MOBINS | City | 16 | 365 (daily) | Human mobility |
| Spain Flowmaps | Regional | 50 | 52 (weekly) | Mobile signal OD |
| Colombia air travel | National | 205 | 52 (weekly) | Air passengers |
| US COVID-19 flows | Macro | 52 | 52 (weekly) | State-level mobility |

Raw data sources and download instructions are provided in `data/README.md`.

---

## Environment

All experiments were run in **Google Colaboratory (Colab Pro)** with a Tesla T4 GPU.

```bash
Python 3.10
PyTorch 2.1.0
NumPy  1.24.4
pandas 2.0.3
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Repository Structure
