# Adjust CA-125 distribution for healthy to reduce % >35 U/mL to ~1–3% (more realistic)
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1000

# Labels
diagnosis = (rng.random(N) < 0.35).astype(int)
ctrl_idx = (diagnosis==0)
case_idx = (diagnosis==1)

# Age
def truncated_normal(mean, sd, low, high, size):
    a = rng.normal(mean, sd, size)
    return np.clip(a, low, high)

age = np.empty(N, dtype=float)
age[ctrl_idx] = truncated_normal(45, 12, 18, 90, ctrl_idx.sum())
age[case_idx] = truncated_normal(58, 10, 18, 90, case_idx.sum())

# Menopause
def sigmoid(x): return 1/(1+np.exp(-x))
p_post = sigmoid((age - 51)/4.5)
menopausal_status = (rng.random(N) < p_post).astype(int)

# BMI
bmi = np.empty(N, dtype=float)
bmi[ctrl_idx] = np.clip(rng.normal(26.5, 5.0, ctrl_idx.sum()), 16, 45)
bmi[case_idx] = np.clip(rng.normal(27.3, 5.2, case_idx.sum()), 16, 45)

# Parity
def truncated_poisson(lmbda, size):
    vals = rng.poisson(lmbda, size)
    return np.clip(vals, 0, 8)
parity = np.empty(N, dtype=int)
parity[ctrl_idx] = truncated_poisson(2.0, ctrl_idx.sum())
parity[case_idx] = truncated_poisson(1.4, case_idx.sum())

# Family history
family_history = np.empty(N, dtype=int)
family_history[ctrl_idx] = (rng.random(ctrl_idx.sum()) < 0.08).astype(int)
family_history[case_idx] = (rng.random(case_idx.sum()) < 0.22).astype(int)

# MHT use
mht_use = np.zeros(N, dtype=int)
post_idx = (menopausal_status==1)
mask_ctrl_post = ctrl_idx & post_idx
mask_case_post = case_idx & post_idx
mht_use[mask_ctrl_post] = (rng.random(mask_ctrl_post.sum()) < 0.20).astype(int)
mht_use[mask_case_post] = (rng.random(mask_case_post.sum()) < 0.25).astype(int)

# BRCA
brca_status = np.zeros(N, dtype=int)
brca_status[ctrl_idx] = (rng.random(ctrl_idx.sum()) < 0.002).astype(int)
case_fh1 = case_idx & (family_history==1)
case_fh0 = case_idx & (family_history==0)
brca_status[case_fh1] = (rng.random(case_fh1.sum()) < 0.40).astype(int)
brca_status[case_fh0] = (rng.random(case_fh0.sum()) < 0.12).astype(int)

# Age shift for BRCA+ cases
age_adjust = (brca_status==1) & case_idx
age[age_adjust] = np.clip(age[age_adjust] - rng.normal(8, 2, age_adjust.sum()), 18, 90)
p_post2 = sigmoid((age - 51)/4.5)
menopause_resample = (rng.random(N) < p_post2).astype(int)
menopausal_status[age_adjust] = menopause_resample[age_adjust]

# Tumor size
tumor_size_cm = np.zeros(N, dtype=float)
ctrl_indices = np.where(ctrl_idx)[0]
ctrl_nonzero_local = (rng.random(ctrl_indices.size) < 0.30)
ctrl_nonzero_idx = ctrl_indices[ctrl_nonzero_local]
tumor_size_cm[ctrl_nonzero_idx] = np.clip(rng.normal(3.0, 1.0, ctrl_nonzero_idx.size), 1.0, 5.0)
sigma_ln = 0.5
median = 9.5
mu_ln = np.log(median)
case_indices = np.where(case_idx)[0]
sizes = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=case_indices.size)
sizes = np.clip(sizes, 0.5, 20.0)
tumor_size_cm[case_indices] = sizes

# CA-125
ca125 = np.zeros(N, dtype=float)

def lognormal_from_median(median, sigma, size):
    mu = np.log(median)
    return rng.lognormal(mean=mu, sigma=sigma, size=size)

# Healthy adjusted baseline: lower median and smaller pre-menopause factor
healthy_base = lognormal_from_median(12.0, 0.50, ctrl_indices.size)
pre_ctrl_mask_local = (menopausal_status[ctrl_indices]==0)
healthy_base[pre_ctrl_mask_local] *= 1.15  # was 1.25
# BMI hemodilution
bmi_ctrl = bmi[ctrl_indices]
hemo_factor = np.exp((bmi_ctrl - 25) * np.log(0.98))
hemo_factor = np.clip(hemo_factor, 0.80, 1.20)
healthy_base *= hemo_factor
# Age mild decrease
age_ctrl = age[ctrl_indices]
age_factor = 1.0 + (-0.002)*(age_ctrl - 45)
age_factor = np.clip(age_factor, 0.85, 1.15)
healthy_base *= age_factor
ca125[ctrl_indices] = np.clip(healthy_base, 2, 2000)

# Cases
case_base = lognormal_from_median(80.0, 0.80, case_indices.size)
beta_size = 0.03
size_effect = np.exp(beta_size*(tumor_size_cm[case_indices] - 9.0))
case_base *= size_effect
bmi_case = bmi[case_indices]
hemo_case_factor = np.exp((bmi_case - 25) * np.log(0.99))
hemo_case_factor = np.clip(hemo_case_factor, 0.85, 1.15)
case_base *= hemo_case_factor
ca125[case_indices] = np.clip(case_base, 2, 2000)

# Rule-based tweaks to tails
healthy_vals = ca125[ctrl_indices]
prop_healthy_high = (healthy_vals > 35).mean()
# If still too high (>0.05), softly pull some values down near 30
if prop_healthy_high > 0.05:
    high_idx = ctrl_indices[ca125[ctrl_indices] > 35]
    pull_n = int(0.5 * high_idx.size)  # pull half down
    if pull_n > 0:
        sel = rng.choice(high_idx, size=pull_n, replace=False)
        ca125[sel] = rng.uniform(20, 34, size=sel.size)

# Ensure small false-positive tail exists (~1–3%)
healthy_vals = ca125[ctrl_indices]
prop_healthy_high = (healthy_vals > 35).mean()
if prop_healthy_high < 0.01:
    bump_n = max(5, int(0.012*ctrl_indices.size))
    bump_sel = rng.choice(ctrl_indices, size=bump_n, replace=False)
    ca125[bump_sel] = np.maximum(ca125[bump_sel], rng.uniform(36, 60, size=bump_sel.size))

# Cases: ensure ≥35% >200 U/mL
case_vals = ca125[case_indices]
prop_case_200 = (case_vals > 200).mean()
if prop_case_200 < 0.35:
    need = int(0.35*case_indices.size - (case_vals > 200).sum())
    if need > 0:
        bump_candidates = case_indices[ca125[case_indices] <= 200]
        if bump_candidates.size > 0:
            sel = rng.choice(bump_candidates, size=min(need, bump_candidates.size), replace=False)
            ca125[sel] = rng.uniform(210, 600, size=sel.size)

# Ultrasound risk
eps = 1e-6
log_ca = np.log(ca125/35.0 + eps)
alpha, gamma_size, gamma_ca = -5.0, 0.22, 0.45
risk_linear = alpha + gamma_size*tumor_size_cm + gamma_ca*log_ca
ultrasound_risk_score = 1/(1+np.exp(-risk_linear))
mask_big_cancer = (diagnosis==1) & (tumor_size_cm>12)
ultrasound_risk_score[mask_big_cancer] = np.maximum(ultrasound_risk_score[mask_big_cancer], 0.70)
mask_no_mass = (diagnosis==0) & (tumor_size_cm==0)
ultrasound_risk_score[mask_no_mass] = np.minimum(ultrasound_risk_score[mask_no_mass], 0.08)
mask_lowrisk_highca = (diagnosis==1) & (ultrasound_risk_score<0.10) & (ca125>200)
ultrasound_risk_score[mask_lowrisk_highca] = 0.30
ultrasound_risk_score = np.clip(ultrasound_risk_score, 0, 1)

# Assemble
df = pd.DataFrame({
    "age": np.round(age).astype(int),
    "menopausal_status": menopausal_status.astype(int),
    "bmi": np.round(bmi, 2),
    "parity": parity.astype(int),
    "family_history": family_history.astype(int),
    "mht_use": mht_use.astype(int),
    "brca_status": brca_status.astype(int),
    "ca125": np.round(ca125, 1),
    "ultrasound_risk_score": np.round(ultrasound_risk_score, 3),
    "tumor_size_cm": np.round(tumor_size_cm, 2),
    "diagnosis_label": diagnosis.astype(int),
})

# BRCA prevalence among cases to 15% target if needed
brca_rate_cases = df.loc[df.diagnosis_label==1, 'brca_status'].mean()
target_brca = 0.15
low, high = 0.10, 0.20
if not (low <= brca_rate_cases <= high):
    case_indices_df = df.index[df.diagnosis_label==1]
    current_pos = int(df.loc[case_indices_df, 'brca_status'].sum())
    target_pos = int(round(target_brca * len(case_indices_df)))
    if target_pos > current_pos:
        to_flip = target_pos - current_pos
        neg_idx = case_indices_df[df.loc[case_indices_df, 'brca_status']==0]
        flip_sel = rng.choice(neg_idx, size=to_flip, replace=False)
        df.loc[flip_sel, 'brca_status'] = 1
        df.loc[flip_sel, 'age'] = np.maximum(18, df.loc[flip_sel, 'age'] - rng.integers(5, 11, size=len(flip_sel)))
        p_post_flip = 1/(1+np.exp(-(df.loc[flip_sel, 'age'] - 51)/4.5))
        df.loc[flip_sel, 'menopausal_status'] = (rng.random(len(flip_sel)) < p_post_flip).astype(int)
    elif target_pos < current_pos:
        to_flip = current_pos - target_pos
        pos_idx = case_indices_df[df.loc[case_indices_df, 'brca_status']==1]
        flip_sel = rng.choice(pos_idx, size=to_flip, replace=False)
        df.loc[flip_sel, 'brca_status'] = 0

# Save CSV
csv_path = "/mnt/data/synthetic_clinical_dataset.csv"
df.to_csv(csv_path, index=False)

# Summaries
summary = {
    "n_rows": int(len(df)),
    "pct_cancer": float(df['diagnosis_label'].mean()),
    "brca_rate_among_cases": float(df.loc[df.diagnosis_label==1, 'brca_status'].mean()),
    "overall_brca_rate": float(df['brca_status'].mean()),
    "healthy_ca125_over_35_pct": float((df.loc[df.diagnosis_label==0, 'ca125']>35).mean()),
    "case_ca125_over_200_pct": float((df.loc[df.diagnosis_label==1, 'ca125']>200).mean()),
    "median_size_cases_cm": float(df.loc[df.diagnosis_label==1, 'tumor_size_cm'].median()),
    "median_size_controls_cm_nonzero": float(df.loc[(df.diagnosis_label==0)&(df.tumor_size_cm>0), 'tumor_size_cm'].median()) if ((df.diagnosis_label==0)&(df.tumor_size_cm>0)).any() else 0.0,
    "ultrasound_risk_mean_cases": float(df.loc[df.diagnosis_label==1, 'ultrasound_risk_score'].mean()),
    "ultrasound_risk_mean_controls": float(df.loc[df.diagnosis_label==0, 'ultrasound_risk_score'].mean()),
}

df.head(10), summary
