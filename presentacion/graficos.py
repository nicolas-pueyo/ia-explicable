"""Genera las figuras SVG de la presentación a partir de modelos.pkl y resultados.json."""
import json, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

HERE = Path(__file__).parent
FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
d = pickle.load(open(HERE / "modelos.pkl", "rb")); M, X, idx = d["models"], d["X"], d["idx"]
R = json.load(open(HERE / "resultados.json"))

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
MODEL = {"tree": "#2a78d6", "lr": "#eb6834", "ebm": "#1baf7a"}
NAME = {"tree": "Árbol", "lr": "Reg. logística", "ebm": "EBM"}
RISK, PROT, NEUTRAL = "#e34948", "#2a78d6", "#f0efec"
CASES = ["A", "B", "C", "D"]
plt.rcParams.update({
    "font.family": "Inter", "font.size": 10, "svg.fonttype": "none",
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.major.size": 0, "ytick.major.size": 0, "figure.facecolor": "none", "axes.facecolor": "none",
})
LBL = {
    "sex_male": "sexo: hombre", "education_high_school": "educ.: secundaria", "education_other": "educ.: otra",
    "education_university": "educ.: universidad", "education_unknown": "educ.: desconocida",
    "marital_status_other_or_unknown": "estado civil: otro", "marital_status_single": "estado civil: soltero",
    "repayment_assistance_plan_yes": "plan de asistencia", "credit_limit": "límite de crédito", "age": "edad",
    "latest_repayment_status": "estado último pago", "maximum_repayment_delay_6m": "retraso máximo 6m",
    "delayed_months_6m": "meses con retraso 6m", "average_bill_6m": "factura media 6m",
    "bill_change_sep_to_apr": "cambio factura", "total_payment_6m": "pago total 6m",
    "payment_to_bill_ratio_6m": "ratio pago/factura", "zero_payment_months_6m": "meses sin pagar 6m",
    "utilization_sep": "utilización (sep)",
}
def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

# ---------- 1. ROC ----------
fig, ax = plt.subplots(figsize=(3.6, 3.1))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (3, 3)))
for k in ["tree", "lr", "ebm"]:
    f, t = R["roc"][k]
    ax.plot(f, t, color=MODEL[k], lw=2, solid_capstyle="round", label=f"{NAME[k]}  {R['cv'][k]['roc_auc'][0]:.3f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
ax.legend(frameon=False, fontsize=8.5, loc="lower right", title="ROC AUC (5-fold)", title_fontsize=8, handlelength=1.2)
ax.set_title("Curva ROC (predicciones out-of-fold)", fontsize=10, loc="left", color=INK, fontweight=600)
save(fig, "roc.svg")

# ---------- 2. Umbral ----------
fig, ax = plt.subplots(figsize=(3.6, 3.1))
for k in ["tree", "lr", "ebm"]:
    th = R["thr"][k]; t = np.array(th["t"]); b = np.array(th["bacc"])
    ax.plot(t, b, color=MODEL[k], lw=2, label=NAME[k])
    i = int(np.argmax(b))
    ax.scatter([t[i]], [b[i]], s=40, color=MODEL[k], edgecolor="white", linewidth=1.5, zorder=5)
ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
th = R["thr"]["ebm"]; t = np.array(th["t"]); b = np.array(th["bacc"]); i = int(np.argmax(b))
ax.annotate(f"EBM: {b[i]:.3f}\numbral {t[i]:.2f}", (t[i], b[i]), xytext=(t[i] + 0.06, b[i] + 0.004),
            fontsize=8.5, color=INK, va="bottom")
j = list(np.round(t, 2)).index(0.5)
ax.annotate(f"umbral 0.5: {b[j]:.3f}", (0.5, b[j]), xytext=(0.53, b[j] + 0.03), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.set_xlabel("Umbral de clasificación"); ax.set_ylabel("Balanced accuracy (CV)")
ax.set_xlim(0.1, 0.7); ax.set_ylim(0.5, 0.73)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", handlelength=1.2)
ax.set_title("Elección del umbral por CV", fontsize=10, loc="left", color=INK, fontweight=600)
save(fig, "umbral.svg")

# ---------- 3. Árbol ----------
tree = M["tree"]; fn = [c.split("__")[1] for c in tree["p"].get_feature_names_out()]
T = tree["m"].tree_
SHORT = {"latest_repayment_status": "estado últ. pago", "payment_to_bill_ratio_6m": "pago/factura",
         "zero_payment_months_6m": "meses sin pagar", "delayed_months_6m": "meses retraso",
         "age": "edad", "bill_change_sep_to_apr": "cambio factura"}
SHORT3 = {"zero_payment_months_6m": "sin pagar", "bill_change_sep_to_apr": "Δ factura", "payment_to_bill_ratio_6m": "pago/fact."}
pos, leaves = {}, []
def walk(n, depth):
    if T.children_left[n] == -1:
        pos[n] = (len(leaves), depth); leaves.append(n); return
    walk(T.children_left[n], depth + 1); walk(T.children_right[n], depth + 1)
    pos[n] = ((pos[T.children_left[n]][0] + pos[T.children_right[n]][0]) / 2, depth)
walk(0, 0)
prob = lambda n: T.value[n][0][1] / T.value[n][0].sum()
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
cmap = LinearSegmentedColormap.from_list("div", [PROT, NEUTRAL, RISK])
norm = TwoSlopeNorm(vmin=0.0, vcenter=R["pos_rate"], vmax=0.8)
Xt = tree["p"].transform(X.loc[idx])
paths = []
for i in range(4):
    n, p = 0, [0]
    while T.children_left[n] != -1:
        n = T.children_left[n] if Xt[i, T.feature[n]] <= T.threshold[n] else T.children_right[n]; p.append(n)
    paths.append(p)
on_path = {}
for i, p in enumerate(paths):
    for a, b in zip(p, p[1:]): on_path.setdefault((a, b), set()).add(i)
fig, ax = plt.subplots(figsize=(12.2, 4.3)); ax.axis("off"); ax.grid(False)
Y = lambda depth: -depth * 1.0
for n in pos:
    if T.children_left[n] == -1: continue
    for c, side in [(T.children_left[n], "≤"), (T.children_right[n], ">")]:
        x0, y0 = pos[n][0], Y(pos[n][1]); x1, y1 = pos[c][0], Y(pos[c][1])
        hot = (n, c) in on_path
        ax.plot([x0, x0, x1, x1], [y0 - 0.18, (y0 + y1) / 2, (y0 + y1) / 2, y1 + 0.2],
                color=INK if hot else "#c9c8c3", lw=2.2 if hot else 1, zorder=2 if hot else 1, solid_joinstyle="round")
for n, (x, depth) in pos.items():
    y = Y(depth); p = prob(n)
    if T.children_left[n] == -1:
        w, h = 0.84, 0.46
        fc = cmap(norm(p)); strong = p >= 0.5
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2 - 0.06), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec=INK if strong else "none", lw=1.6 if strong else 0, zorder=3))
        ax.text(x, y + 0.02, f"{p:.0%}", ha="center", va="center", fontsize=10, fontweight=700,
                color="white" if (p > 0.55 or p < 0.08) else INK, zorder=4)
        ax.text(x, y - 0.19, f"n={T.n_node_samples[n]}", ha="center", va="center", fontsize=7,
                color="white" if (p > 0.55 or p < 0.08) else INK2, zorder=4)
    else:
        f = fn[T.feature[n]]; thr = T.threshold[n]
        name = SHORT.get(f, f) if depth < 3 else SHORT3.get(f, SHORT.get(f, f))
        txt = f"{name} ≤ {thr:g}" if abs(thr) < 100 else f"{name} ≤ {thr:,.0f}"
        ax.text(x, y, txt, ha="center", va="center", fontsize=9 if depth < 3 else 7.8, color=INK,
                fontweight=600 if depth == 0 else 500, zorder=4,
                bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.6", fc="white", ec=GRID, lw=0.8))
for n in set(p[-1] for p in paths):
    who = [CASES[i] for i, p in enumerate(paths) if p[-1] == n]
    x, depth = pos[n]
    for k, c in enumerate(who):
        cx = x + (k - (len(who) - 1) / 2) * 0.34
        ax.add_patch(plt.Circle((cx, Y(depth) - 0.62), 0.14, fc=INK, ec="none", zorder=5))
        ax.text(cx, Y(depth) - 0.62, c, ha="center", va="center", color="white", fontsize=8.5, fontweight=700, zorder=6)
ax.text(-0.5, 0.12, "izquierda = cumple (≤)   ·   derecha = no cumple (>)", fontsize=8, color=MUTED, va="bottom")
ax.set_xlim(-0.55, len(leaves) - 0.45); ax.set_ylim(Y(4) - 0.85, 0.35); ax.set_aspect("auto")
save(fig, "arbol.svg")

# ---------- local helpers ----------
def local_panel(ax, contrib, base, p, title, nshow=5, xlim=None):
    s = pd.Series(contrib)
    top = s.reindex(s.abs().sort_values(ascending=False).index)[:nshow]
    rest = s.drop(top.index).sum()
    items = list(top.items()) + [("resto de variables", rest)]
    items = items[::-1]
    yy = np.arange(len(items))
    vals = [v for _, v in items]
    ax.barh(yy, vals, color=[RISK if v > 0 else PROT for v in vals], height=0.62, edgecolor="white", linewidth=1)
    ax.set_yticks(yy); ax.set_yticklabels([LBL.get(k, k) for k, _ in items], fontsize=8)
    for y_, v in zip(yy, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), y_, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
    if xlim: ax.set_xlim(*xlim)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_title(title, fontsize=9, loc="left", color=INK, fontweight=600)
    ax.text(1.0, 1.02, f"P = {p:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            fontweight=700, color=RISK if p >= 0.5 else PROT)

TITLES = [f"{c} · real {y}" for c, y in zip(CASES, R["local_y"])]

# ---------- 4. LR coeficientes ----------
lr = M["lr"]; fl = [c.split("__")[1] for c in lr["p"].get_feature_names_out()]; co = lr["m"].coef_[0]
s = pd.Series(co, index=fl).sort_values()
fig, ax = plt.subplots(figsize=(4.3, 6.9))
ax.barh(range(len(s)), s.values, color=[RISK if v > 0 else PROT for v in s.values], height=0.66,
        edgecolor="white", linewidth=1)
ax.set_yticks(range(len(s))); ax.set_yticklabels([LBL[k] for k in s.index], fontsize=8.5)
for i, v in enumerate(s.values):
    ax.text(v + (0.012 if v >= 0 else -0.012), i, f"{v:+.2f}  (OR {np.exp(v):.2f})" if abs(v) > 0.1 else f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=7.5, color=INK2)
ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
ax.set_xlim(-0.42, 0.85); ax.set_xlabel("Coeficiente (log-odds por 1 desv. típica)")
i_md = list(s.index).index("maximum_repayment_delay_6m")
ax.annotate("signo invertido por\ncolinealidad (r = 0,82)", (s.iloc[i_md] + 0.01, i_md), xytext=(0.12, i_md + 0.6),
            fontsize=7.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
save(fig, "lr_coef.svg")

# ---------- 5. LR local ----------
Xl = lr["p"].transform(X.loc[idx])
fig, axs = plt.subplots(2, 2, figsize=(5.8, 5.6), sharex=True)
for i, ax in enumerate(axs.flat):
    local_panel(ax, dict(zip(fl, Xl[i] * co)), lr["m"].intercept_[0], R["local_p"]["lr"][i], TITLES[i], 5, (-1.25, 2.1))
for ax in axs[1]: ax.set_xlabel("contribución al log-odds", fontsize=8)
fig.tight_layout(w_pad=0.8, h_pad=1.0)
save(fig, "lr_local.svg")

# ---------- 6. EBM formas ----------
ebm = M["ebm"]["m"]
def shape(j):
    b = ebm.bins_[j][0]; sc = ebm.term_scores_[j][1:-1]
    return np.asarray(b), sc
fig, axs = plt.subplots(2, 2, figsize=(5.4, 3.9))
spec = [(16, "ratio pago/factura", (0, 3)), (10, "estado último pago", None),
        (17, "meses sin pagar (de 6)", None), (9, "edad", (21, 70))]
imp = dict(zip(range(len(fn)), ebm.term_importances()))
for ax, (j, title, xl) in zip(axs.flat, spec):
    b, sc = shape(j)
    if len(b) < 15:   # variable discreta: escalones centrados en el valor entero
        vals = np.r_[b[0] - 0.5, (b[:-1] + b[1:]) / 2, b[-1] + 0.5]
        ax.bar(vals, sc, width=0.72, color=[RISK if v > 0 else PROT for v in sc], edgecolor="white", linewidth=1)
    else:
        edges = np.r_[b[0] - (b[1] - b[0]), b, b[-1] + (b[-1] - b[-2])]
        xs = np.repeat(edges, 2)[1:-1]; ys = np.repeat(sc, 2)
        ax.fill_between(xs, 0, ys, where=ys > 0, color=RISK, alpha=0.18, lw=0, interpolate=True)
        ax.fill_between(xs, 0, ys, where=ys < 0, color=PROT, alpha=0.18, lw=0, interpolate=True)
        ax.plot(xs, ys, color=INK, lw=1.6)
    ax.axhline(0, color=INK2, lw=0.8)
    if xl: ax.set_xlim(*xl)
    ax.set_title(f"{title}", fontsize=8.8, loc="left", color=INK, fontweight=600)
    ax.text(1, 1.02, f"imp. {imp[j]:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.tick_params(labelsize=7.5)
for ax in axs[:, 0]: ax.set_ylabel("log-odds", fontsize=8)
fig.tight_layout(w_pad=1.0, h_pad=1.2)
save(fig, "ebm_formas.svg")

# ---------- 7. EBM importancias ----------
s = pd.Series(ebm.term_importances(), index=[LBL[f] for f in fn]).sort_values().tail(8)
fig, ax = plt.subplots(figsize=(2.6, 2.5))
ax.barh(range(len(s)), s.values, color=MODEL["ebm"], height=0.62, edgecolor="white")
ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
for i, v in enumerate(s.values): ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=7.2, color=INK2)
ax.grid(axis="y", visible=False); ax.tick_params(axis="x", labelsize=7.5); ax.set_xlim(0, 0.39)
ax.set_xlabel("importancia media |log-odds|", fontsize=8)
save(fig, "ebm_imp.svg")

# ---------- 8. EBM local ----------
Xe = M["ebm"]["p"].transform(X.loc[idx]); terms = ebm.eval_terms(Xe)
fig, axs = plt.subplots(2, 2, figsize=(5.8, 4.5), sharex=True)
for i, ax in enumerate(axs.flat):
    local_panel(ax, dict(zip(fn, terms[i])), ebm.intercept_[0], R["local_p"]["ebm"][i], TITLES[i], 5, (-1.05, 1.55))
for ax in axs[1]: ax.set_xlabel("contribución al log-odds", fontsize=8)
fig.tight_layout(w_pad=0.8, h_pad=1.0)
save(fig, "ebm_local.svg")

# ---------- 9. Comparación local (3 modelos x 4 casos) ----------
fig, ax = plt.subplots(figsize=(4.0, 2.3))
ax.axvspan(0.5, 1, color=RISK, alpha=0.06, lw=0)
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
for i in range(4):
    ps = [R["local_p"][k][i] for k in ["tree", "lr", "ebm"]]
    ax.plot([min(ps), max(ps)], [3 - i] * 2, color="#c9c8c3", lw=2.5, solid_capstyle="round", zorder=1)
    for k, off in zip(["tree", "lr", "ebm"], [0, 0, 0]):
        ax.scatter(R["local_p"][k][i], 3 - i, s=70, color=MODEL[k], edgecolor="white", linewidth=1.6, zorder=3,
                   label=NAME[k] if i == 0 else None)
ax.set_yticks(range(4)); ax.set_yticklabels(TITLES[::-1], fontsize=8.5)
ax.set_xlim(0, 0.8); ax.set_xlabel("P(dificultad de pago)", fontsize=8.5); ax.grid(axis="y", visible=False)
ax.text(0.51, 3.45, "predice 1", fontsize=7.5, color=RISK, va="bottom")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, handletextpad=0.2)
ax.set_ylim(-0.5, 3.75); ax.tick_params(axis="x", labelsize=7.5)
save(fig, "local_comparacion.svg")
print("ok")
