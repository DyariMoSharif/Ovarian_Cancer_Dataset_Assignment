# Synthetic Ovarian Cancer Dataset (Fully Synthetic, with CTGAN Extension)

> **Author:** Dyari Mohammed Sharif  
> **Goal:** Provide a clinically plausible, fully synthetic cohort (≈1,000 women) for training and benchmarking early–stage ovarian cancer risk models. No real patient data is used.  

---

## Overview

This repository documents both the **hand-crafted synthetic schema** and a **CTGAN-based generative pipeline** for producing realistic ovarian cancer datasets.  
- The schema enforces **clinical plausibility** (ranges, correlations, overlaps) and avoids leakage.  
- The CTGAN model augments this by learning **complex joint distributions** from the schema-based dataset and producing high-fidelity tabular data.  

---

## Key Principles

- **Clinical plausibility.** Feature ranges/distributions are grounded in epidemiology and practice.  
- **Realistic overlap.** Early cases overlap with controls (e.g., CA-125 near normal).  
- **Dependencies encoded.** Age drives menopause, family history drives BRCA probability, BMI affects CA-125 in controls.  
- **No leakage.** Class-conditional feature generation avoids trivial signal.  
- **ML-readiness.** Distributions bounded, skewed as appropriate.  
- **Generative refinement.** A **CTGAN pipeline** (from the [SDV library](https://github.com/sdv-dev/CTGAN)) provides an additional generative layer with statistical quality evaluation.  

---

## Specification Table (Schema & Constraints)

| Feature | Type / Units | Target Range / Values | Medical realism & constraints (correlations encoded) |
|---|---|---|---|
| `age` | integer (years) | 18–90 (truncate tails) | Cases skew older. Tie `menopausal_status` tightly to age (median natural menopause ≈ 51 y; steep rise after ~50–55). |
| `menopausal_status` | binary | 0 = pre, 1 = post | Sample from age; pre-menopause allows benign CA-125 elevations; post-menopause commonly uses 35 U/mL cut-off in practice. |
| `bmi` | continuous (kg/m²) | 16–45 (winsorize) | Weak main effect on risk; enforce interactions: if `mht_use=1` **or** `family_history=1` then BMI→risk ≈ 0. Apply hemodilution on CA-125 in controls (≈2–3% ↓ per BMI unit, capped at ~20%). |
| `parity` | integer (# births) | 0–8 | Protective: multiply odds by ~0.87 per birth; keep approx. log-linear up to ~5; nulliparity highest risk. |
| `family_history` | binary | 0 = no, 1 = yes (first-degree breast/ovarian) | Enrich among cases. Attenuates BMI effect (BMI→risk ≈ 0 if FH=1). Higher P(BRCA+ \| FH=1) in cases. |
| `mht_use` | binary | 0 = never, 1 = ever (systemic MHT) | Mostly post-menopause; modest positive main effect (OR ~1.2–1.4). Interaction gate: if `mht_use=1`, set BMI→risk ≈ 0. |
| `brca_status` | binary | 0 = negative, 1 = positive | Rare in controls; enriched in cases (~15–20% of EOC). BRCA+ cases skew younger at diagnosis by ~6–10 y; raise P(BRCA+ \| FH=1). |
| `ca125` | continuous (U/mL) | 2–2,000 (right-skew) | Log-normal with deliberate overlap. Early-stage sensitivity limited (~23–50%) → many early cancers near/≤35. In controls: CA-125 tends lower with age and higher BMI (hemodilution). In cases: add small + slope vs `tumor_size_cm`. |
| `ultrasound_risk_score` | continuous probability | 0.00–1.00 | ADNEX-like probability primarily from `tumor_size_cm` (key driver) plus a small CA-125 term; map to O-RADS bands; maintain only moderate correlation with CA-125 (ρ≈0.3–0.5). |
| `tumor_size_cm` | continuous (cm) | Controls: 0 or simple cysts ~1–5; Cases: 0.5–20 (right-skew, center ~8–11) | Allow benign simple cysts in controls; for early cancers anchor size distribution and maintain a modest + correlation with CA-125 in cases. |
| `diagnosis_label` | binary | 0 = healthy, 1 = early-stage OC | Supervised target; all upstream variables are class-conditional to avoid leakage. |

---

## Feature Priors & Distributional Families (Typical Parameters)

| Feature | Unit | Healthy Distribution | Cancer Distribution | Notes |
|---|---|---|---|---|
| `age` | years | Trunc. Normal 18–90; mean ~45, sd ~12 | Trunc. Normal 18–90; mean ~58, sd ~10 | Age drives menopause; cases skew older. |
| `menopausal_status` | 0/1 | Bernoulli conditioned on age (e.g., P(post ≥55) ≈ 0.85) | Same rule (by age) | Gates biomarker strata and MHT eligibility. |
| `bmi` | kg/m² | Trunc. Normal 16–45; mean ~26.5, sd ~5.0 | Trunc. Normal 16–45; mean ~27.3, sd ~5.2 | Weak main effect; interaction gates with `mht_use` and `family_history`; hemodilution on CA-125 in controls. |
| `parity` | births | Trunc. Poisson (λ≈2), 0–8 | Trunc. Poisson (λ≈1.4), 0–8 | Protective (~6–13% odds ↓ per birth). |
| `family_history` | 0/1 | Bernoulli p≈0.08 | Bernoulli p≈0.22 | Major risk factor; raises P(BRCA+ \| FH=1). |
| `mht_use` | 0/1 | Bernoulli among post-menopause (e.g., p≈0.20); ~0 pre-menopause | Bernoulli among post-menopause (e.g., p≈0.25) | Modest + main effect (OR ~1.2–1.4); BMI effect ≈ 0 if `mht_use=1`. |
| `brca_status` | 0/1 | Bernoulli p≈0.002 | Bernoulli p≈0.12–0.18 | Enriched in cases; BRCA+ cases younger by ~6–10 y. |
| `ca125` | U/mL | Log-normal; median ~12–20; ~1% >35; range 2–2000 | Log-normal; early-stage median ~60–120; heavy right tail; 2–2000 | Menopause-aware overlap; benign elevations allowed in controls. |
| `ultrasound_risk_score` | 0.0–1.0 | Probability from logistic head primarily on size; typical near 0–0.1 (O-RADS 2–3) | Same model; broader 0.2–0.9 (O-RADS 4–5) | Keep ρ(CA-125, score) ≈ 0.3–0.5; size is main driver. |
| `tumor_size_cm` | cm | Zero-inflated: mass=0 (no lesion) w/ high prob; else benign simple cyst ~1–5 (Trunc. Normal μ≈3, σ≈1) | Log-normal (right-skew), support 0.5–20; center ~8–11 | Positive (modest) correlation with CA-125 in cases. |
| `diagnosis_label` | 0/1 | By definition 0 | By definition 1 | Labels assigned by construction. |

---

## CTGAN Extension

We incorporate **CTGAN (Conditional Tabular GAN)** to further synthesize datasets that mimic the statistical properties of the base synthetic cohort.

### Why CTGAN?
- Vanilla GANs struggle with **imbalanced tabular data** and mixed **continuous + categorical features**.  
- CTGAN addresses this by conditionally sampling discrete columns and better capturing **multi-modal distributions**.  

### Pipeline
1. **Input data**: schema-generated synthetic dataset (`data/synthetic_clinical_dataset.csv`, 1,000 rows × 11 features).  
2. **Model**: CTGAN trained for **3,000 epochs** with categorical conditioning (`menopausal_status, parity, family_history, mht_use, brca_status, diagnosis_label`).  
3. **Generation**: produces new synthetic cohort of 1,000 rows (`data/ctGAN_synthetic_data_generated.csv`).  
4. **Evaluation**: quality assessed with SDV’s `evaluate_quality`, producing:  
   - **Column Shapes Score:** ~88%  
   - **Column Pair Trends Score:** ~86%  
   - **Overall Quality Score:** ~87%  

### Example Metric Snapshot
| Feature               | Metric        | Score |
|------------------------|--------------|-------|
| Age                   | KSComplement | 0.919 |
| Menopausal status     | TVComplement | 0.872 |
| BMI                   | KSComplement | 0.925 |
| Parity                | TVComplement | 0.914 |
| Family history        | TVComplement | 0.918 |
| MHT use               | TVComplement | 0.867 |
| BRCA status           | TVComplement | 0.931 |
| CA-125                | KSComplement | 0.846 |
| Ultrasound risk score | KSComplement | 0.839 |
| Tumor size (cm)       | KSComplement | 0.774 |
| Diagnosis label       | TVComplement | 0.891 |

---

## Loss Function Behavior

- **Generator loss** typically stabilizes at a **negative value** after many epochs.  
- **Discriminator loss** oscillates around **zero**.  
- These patterns are consistent with CTGAN benchmarks and indicate convergence.  

---

## Usage

- **Schema-only synthetic dataset**: deterministic, epidemiologically grounded.  
- **CTGAN-augmented dataset**: stochastic generation with distributional fidelity metrics.  
- Users can choose between **rule-based realism** and **GAN-based generative diversity**, or use both for robustness.  
