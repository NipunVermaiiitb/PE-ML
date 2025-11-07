# ======================================================
# 🔥 Final Corrected Subject-level Stress/Recovery Indices (log-based SI)
# ======================================================

!pip install -q scikit-learn pandas numpy matplotlib seaborn

import os, json, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# ====== CONFIG ======
csv_path = "/content/All_subjects_segments60.csv"  # change if needed
outdir = "/content/si_ri_model_outputs_logfix"
os.makedirs(outdir, exist_ok=True)

# ======================================================
# 1) Load & robust preprocessing
# ======================================================
df = pd.read_csv(csv_path)
print("✅ Loaded:", df.shape)

for c in ["scope", "segment", "Subject_Name", "ApEn"]:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

def norm_cond(x):
    lx = str(x).strip().lower()
    if lx in {"baseline","base"}: return "Baseline"
    if lx in {"stroop","mat","stress"}: return "Stress"
    if lx in {"recovery","post","post_stress"}: return "Recovery"
    return x
df["condition"] = df["condition"].apply(norm_cond)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != "Subject_ID"]

# winsorize 1/99
q_low, q_hi = df[num_cols].quantile(0.01), df[num_cols].quantile(0.99)
for c in num_cols:
    df[c] = np.clip(df[c], q_low[c], q_hi[c])

# log-transform skewed positives
sk = df[num_cols].skew(numeric_only=True)
for c in num_cols:
    if (df[c] > 0).all() and abs(sk[c]) > 1.0:
        df[c] = np.log1p(df[c])

# impute + scale
imp = IterativeImputer(random_state=42, max_iter=15)
scaler = RobustScaler()
df[num_cols] = imp.fit_transform(df[num_cols])
df[num_cols] = scaler.fit_transform(df[num_cols])

# ======================================================
# 2) Aggregate per subject/condition
# ======================================================
agg = df.groupby(["Subject_ID", "condition"], as_index=False)[num_cols].mean()
wide = agg.pivot_table(index="Subject_ID", columns="condition", values=num_cols)
wide.columns = [f"{f}__{c}" for f,c in wide.columns.to_flat_index()]
wide.reset_index(inplace=True)

def have_all(f):
    return all(f"{f}__{p}" in wide.columns for p in ["Baseline","Stress","Recovery"])
base_feats = [f for f in {c.split("__")[0] for c in wide.columns} if have_all(f)]

# ======================================================
# 3) Compute indices (log-based SI + standard RI)
# ======================================================
eps = 1e-9
for f in base_feats:
    B, S, R = wide[f"{f}__Baseline"], wide[f"{f}__Stress"], wide[f"{f}__Recovery"]
    # Log-based Stress Index
    wide[f"SI_{f}"] = np.abs(np.log1p(np.abs(S)) - np.log1p(np.abs(B)))
    # Recovery Index same as before
    wide[f"RI_{f}"] = ((R - S) / (B - S + eps)).clip(0,1)

wide["SI_true"] = wide[[f"SI_{f}" for f in base_feats]].mean(axis=1)
wide["RI_true"] = wide[[f"RI_{f}" for f in base_feats]].mean(axis=1)

# ======================================================
# 4) Build feature matrix (levels + deltas)
# ======================================================
X_parts = {}
for f in base_feats:
    B, S, R = wide[f"{f}__Baseline"], wide[f"{f}__Stress"], wide[f"{f}__Recovery"]
    X_parts[f"{f}__Baseline"] = B
    X_parts[f"{f}__Stress"] = S
    X_parts[f"{f}__Recovery"] = R
    X_parts[f"{f}__SminusB"] = S - B
    X_parts[f"{f}__RminusS"] = R - S
    X_parts[f"{f}__RminusB"] = R - B
X = pd.DataFrame(X_parts).replace([np.inf, -np.inf], np.nan).fillna(0)

y_SI = wide["SI_true"].values
y_RI = wide["RI_true"].values
subjects = wide["Subject_ID"].values

# ======================================================
# 5) Train/test split by subject
# ======================================================
Xtr, Xte, ySItr, ySIte, yRItr, yRIte, subtr, subte = train_test_split(
    X, y_SI, y_RI, subjects, test_size=0.3, random_state=42
)

# ======================================================
# 6) Train models (with feature importance selection)
# ======================================================
def fit_rf(Xtr, ytr, top_k=20, seed=42):
    rf = RandomForestRegressor(n_estimators=500, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    keep = imp.head(min(top_k, len(imp))).index
    final = RandomForestRegressor(n_estimators=800, random_state=seed, n_jobs=-1)
    final.fit(Xtr[keep], ytr)
    return final, keep, imp

model_SI, keep_SI, imp_SI = fit_rf(Xtr, ySItr, top_k=25, seed=11)
model_RI, keep_RI, imp_RI = fit_rf(Xtr, yRItr, top_k=25, seed=22)

SI_pred = model_SI.predict(Xte[keep_SI])
RI_pred = model_RI.predict(Xte[keep_RI])

# ======================================================
# 7) Metrics & plots
# ======================================================
def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pr, pp = pearsonr(y_true, y_pred) if len(y_true) > 1 else (np.nan, np.nan)
    return dict(R2=r2, MAE=mae, RMSE=rmse, Pearson_r=pr, p_value=pp)

m_SI = metrics(ySIte, SI_pred)
m_RI = metrics(yRIte, RI_pred)

def scatter_plot(y_true, y_pred, title, path):
    plt.figure(figsize=(5,5))
    sns.scatterplot(x=y_true, y=y_pred)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lim, lim, '--k')
    plt.xlabel("True"); plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=160)
    plt.show()

scatter_plot(ySIte, SI_pred, f"Stress Index (log-based)\nR²={m_SI['R2']:.3f}, r={m_SI['Pearson_r']:.3f}", f"{outdir}/si_true_vs_pred.png")
scatter_plot(yRIte, RI_pred, f"Recovery Index\nR²={m_RI['R2']:.3f}, r={m_RI['Pearson_r']:.3f}", f"{outdir}/ri_true_vs_pred.png")

# Feature importances
def plot_imp(imp, title, path, top=15):
    topv = imp.head(top)[::-1]
    plt.figure(figsize=(6,4))
    plt.barh(topv.index, topv.values)
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=160)
    plt.show()

plot_imp(imp_SI, "Top Features for Stress Index (log-based)", f"{outdir}/si_feature_importance.png")
plot_imp(imp_RI, "Top Features for Recovery Index", f"{outdir}/ri_feature_importance.png")

# ======================================================
# 8) Save results (JSON-safe fix)
# ======================================================

# Convert Index objects to lists so JSON can serialize them
summary = {
    "SI_metrics": m_SI,
    "RI_metrics": m_RI,
    "n_subjects_total": int(len(subjects)),
    "selected_features_SI": list(keep_SI),
    "selected_features_RI": list(keep_RI)
}

# Save JSONs
with open(f"{outdir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
with open(f"{outdir}/upload_me_for_analysis.json", "w") as f:
    json.dump(summary, f, indent=2)

# ZIP everything
zip_path = "/content/si_ri_model_outputs_logfix.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(outdir):
        for fn in files:
            z.write(os.path.join(root, fn), os.path.relpath(os.path.join(root, fn), outdir))

print("\n✅ Finished successfully.")
print("ZIP saved at:", zip_path)

