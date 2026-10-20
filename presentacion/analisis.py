"""Reproduce los modelos del notebook (Fase II V2) y guarda los resultados para las slides."""
import json, pickle
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import roc_curve, accuracy_score, balanced_accuracy_score, recall_score, precision_score
from interpret.glassbox import ExplainableBoostingClassifier

HERE = Path(__file__).parent
train = pd.read_csv(HERE.parent / "data/train.csv")
y = train["payment_difficulty_next_month"]
X = train.drop(columns=["payment_difficulty_next_month", "row_id"])
cat = ["sex", "education", "marital_status", "repayment_assistance_plan"]
num = [c for c in X.columns if c not in cat]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def prep(scale):
    return ColumnTransformer([("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
                              ("num", StandardScaler() if scale else "passthrough", num)])

models = {
    "tree": Pipeline([("p", prep(False)), ("m", DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=100, random_state=42))]),
    "lr": Pipeline([("p", prep(True)), ("m", LogisticRegression(C=0.1, max_iter=1000, random_state=42))]),
    "ebm": Pipeline([("p", prep(False)), ("m", ExplainableBoostingClassifier(interactions=0, random_state=42))]),
}
out = {"n": len(y), "pos_rate": float(y.mean()), "cv": {}, "roc": {}, "thr": {}}
oof = {}
for k, pipe in models.items():
    r = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "balanced_accuracy", "recall", "roc_auc"], n_jobs=-1)
    out["cv"][k] = {m: [float(r[f"test_{m}"].mean()), float(r[f"test_{m}"].std())] for m in ["accuracy", "balanced_accuracy", "recall", "roc_auc"]}
    p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof[k] = p
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.linspace(0, len(fpr) - 1, 300).astype(int)
    out["roc"][k] = [fpr[idx].tolist(), tpr[idx].tolist()]
    ts = np.round(np.arange(0.10, 0.71, 0.01), 2)
    out["thr"][k] = {"t": ts.tolist(),
                     "acc": [accuracy_score(y, p >= t) for t in ts],
                     "bacc": [balanced_accuracy_score(y, p >= t) for t in ts],
                     "rec": [recall_score(y, p >= t) for t in ts],
                     "prec": [precision_score(y, p >= t, zero_division=0) for t in ts]}
    pipe.fit(X, y)
    print(k, out["cv"][k])

# Instancias locales (mismas que el notebook: primera TN/FP/TP/FN del árbol sobre train)
yp = models["tree"].predict(X)
idx = [X[(yp == a) & (y == b)].index[0] for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]]
out["local_idx"] = [int(i) for i in idx]
out["local_rowid"] = train.loc[idx, "row_id"].tolist()
out["local_y"] = y.loc[idx].tolist()
out["local_X"] = X.loc[idx].astype(object).where(pd.notna(X.loc[idx]), None).values.tolist()
out["cols"] = X.columns.tolist()
out["local_p"] = {k: m.predict_proba(X.loc[idx])[:, 1].tolist() for k, m in models.items()}
pickle.dump({"models": models, "X": X, "y": y, "idx": idx, "oof": oof}, open(HERE / "modelos.pkl", "wb"))
json.dump(out, open(HERE / "resultados.json", "w"), indent=1, default=str)
