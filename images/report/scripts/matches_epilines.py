"""Fig. fig:matches: correspondencias inliers/outliers e linhas epipolares do par SIFT img_0774__img_0790.

Recalcula o par do zero com o mesmo pipeline do repositorio (so leitura em R):
  src/detectores.py         SIFT_create(nfeatures=5000) em tons de cinza, resolucao cheia 1920x1080
  src/matching/matcher.py   BFMatcher NORM_L2, crossCheck=False, knnMatch(k=2)
  src/matching/filters.py   Lowe 0,75 (config_cozinha.yaml filters.lowe_ratio)
  src/matching/cross_check  vizinho mutuo apos Lowe (filters.cross_check: true)
  src/epipolar/estimate.py  findFundamentalMat e findEssentialMat USAC_MAGSAC 1 px, 0,999, 5000 it.;
                            inlier = aceito por F E por E
e confere contagens e mascara contra cozinha/results/config_cozinha e matching_metrics.csv.
"""
import csv
import sys

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

R = "/home/moliveira/Documents/3d_indoor_environment"
sys.path.insert(0, R)
from src.epipolar import estimate as epi  # noqa: E402
from src.matching import cross_check, filters, matcher  # noqa: E402

A, B = "img_0774", "img_0790"
PAIR = f"{A}__{B}"
ART = f"{R}/cozinha/artifacts/config_cozinha"
RES = f"{R}/cozinha/results/config_cozinha"
OUT = "/home/moliveira/Documents/report_3d_indoor_environment/images/report/matches_epilines.jpg"
CROP = (330, 990)         # faixa vertical exibida (linhas y0:y1); a cena util (bancada) fica abaixo do azulejo
N_IN, N_OUT, N_EPI = 60, 30, 10

cfg = yaml.safe_load(open(f"{R}/config_cozinha.yaml"))
gray = [cv2.imread(f"{R}/cozinha/frames/all_stride10/{f}.png", cv2.IMREAD_GRAYSCALE) for f in (A, B)]
rgb = [cv2.cvtColor(cv2.imread(f"{R}/cozinha/frames/all_stride10/{f}.png"), cv2.COLOR_BGR2RGB) for f in (A, B)]

# 1) deteccao (src/detectores.py)
sift = cv2.SIFT_create(nfeatures=cfg["detection"]["nfeatures"])
(kpa, da), (kpb, db) = [sift.detectAndCompute(g, None) for g in gray]
# 2) matching + Lowe + cross check
knn = matcher.run(cfg["matching"], da).knnMatch(da, db, k=2)
lowe = filters.run(cfg["filters"], knn).matches
lowe_arr = np.array([(m.queryIdx, m.trainIdx, m.distance) for m in lowe], np.float32)
cross = cross_check.mutual(lowe_arr, da, db, cfg["matching"])
qa, tb = cross[:, 0].astype(int), cross[:, 1].astype(int)
pa = np.float32([kpa[i].pt for i in qa])
pb = np.float32([kpb[i].pt for i in tb])
# 3) geometria (src/epipolar/estimate.py)
K = np.array([[cfg["geometry"]["intrinsics"]["fx"], 0, cfg["geometry"]["intrinsics"]["cx"]],
              [0, cfg["geometry"]["intrinsics"]["fy"], cfg["geometry"]["intrinsics"]["cy"]], [0, 0, 1]], float)
res = epi.estimate(cfg["geometry"], pa, pb, K)
F, inl = res.fundamental, res.inlier_mask

# conferencia com o pipeline salvo
row = next(r for r in csv.DictReader(open(f"{ART}/matching_metrics.csv")) if r["detector"] == "sift" and r["pair"] == PAIR)
saved_cross = np.load(f"{RES}/matching/sift/{PAIR}/matches_cross.npy")
saved_mask = np.load(f"{RES}/epipolar/sift/{PAIR}/inlier_mask.npy")
print(f"raw {len(knn)} (csv {row['n_matches_raw']}) | lowe {len(lowe)} | cross {len(cross)} (csv {row['n_matches_lowe']})"
      f" | inliers {inl.sum()} (csv {row['inliers']}) | razao {inl.mean():.4f} (csv {float(row['inlier_ratio']):.4f})"
      f" | sampson med {np.mean(res.sampson_errors):.4f} (csv {float(row['sampson_px']):.4f})")
# sampson_px do CSV vem de src/relatorio.py:sampson: media sobre os inliers com F = K^-T E K^-1 (E do MAGSAC)
Ki = np.linalg.inv(K)
Fe = Ki.T @ res.essential @ Ki
x1, x2 = np.c_[pa[inl], np.ones(inl.sum())], np.c_[pb[inl], np.ones(inl.sum())]
Fx1, Ftx2 = x1 @ Fe.T, x2 @ Fe
samp_e = np.sqrt((x2 * Fx1).sum(1) ** 2 / (Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2)).mean()
print(f"sampson (E, como relatorio.py) {samp_e:.4f} px (csv {float(row['sampson_px']):.4f})")
same_matches = np.array_equal(cross[:, :2], saved_cross[:, :2])
print("matches identicos aos salvos:", same_matches,
      "| mascara identica:", same_matches and np.array_equal(inl, saved_mask),
      "| concordancia mascara:", (inl == saved_mask).mean() if same_matches else "n/a")

# ---------------------------------------------------------------- figura
y0, y1 = CROP
H = y1 - y0
W = 1920
GAP = 40  # faixa branca entre A (cima) e B (baixo), em px da imagem


def stack(a, b):
    return np.vstack([a[y0:y1], np.full((GAP, W, 3), 255, np.uint8), b[y0:y1]])


def in_crop(p):
    return (p[:, 1] >= y0) & (p[:, 1] < y1)


def to_a(p):   # coordenadas do quadro A no painel empilhado
    return np.c_[p[:, 0], p[:, 1] - y0]


def to_b(p):   # coordenadas do quadro B no painel empilhado
    return np.c_[p[:, 0], p[:, 1] - y0 + H + GAP]


rng = np.random.default_rng(0)
vis = in_crop(pa) & in_crop(pb)
idx_in = np.flatnonzero(inl & vis)
idx_out = np.flatnonzero(~inl & vis)
# amostra uniforme no espaco: ordena por x de A e pega passos iguais
idx_in_s = idx_in[np.argsort(pa[idx_in, 0])][np.linspace(0, len(idx_in) - 1, min(N_IN, len(idx_in))).astype(int)]
idx_out_s = rng.choice(idx_out, min(N_OUT, len(idx_out)), replace=False)

# epipolares: N_EPI inliers espalhados (farthest point sampling em A, so pontos dentro do recorte)
MARG = 60  # px: pontos longe das bordas do recorte nos dois quadros
inner = lambda p: (p[:, 0] > MARG) & (p[:, 0] < W - MARG) & (p[:, 1] > y0 + MARG) & (p[:, 1] < y1 - MARG)
cand = idx_in[inner(pa[idx_in]) & inner(pb[idx_in])]
# retas separadas: altura da reta epipolar em B no centro (x = W/2) dividida em N_EPI faixas iguais;
# em cada faixa, o ponto mais proximo de um x alvo que varre a largura (pontos e retas espalhados)
lb = (F @ np.c_[pa[cand], np.ones(len(cand))].T).T
ymid = -(lb[:, 0] * W / 2 + lb[:, 2]) / lb[:, 1]
edges = np.linspace(ymid.min(), ymid.max(), N_EPI + 1)
xt = np.r_[0.12, 0.62, 0.32, 0.85, 0.22, 0.5, 0.75, 0.15, 0.4, 0.9][:N_EPI] * W
sel = []
for k in range(N_EPI):
    m = np.flatnonzero((ymid >= edges[k]) & (ymid <= edges[k + 1]))
    if sel:  # retas a pelo menos 40 px uma da outra no centro de B
        m = m[np.min(np.abs(ymid[m][:, None] - ymid[np.searchsorted(cand, sel)][None]), axis=1) >= 40]
    if len(m):
        sel.append(cand[m[np.argmin(np.abs(pa[cand[m], 0] - xt[k]))]])
sel = np.array(sel)
cols = [plt.cm.tab10(i) for i in range(10)]


def clip_line(l, ytop, ybot):
    """Segmento da reta a x + b y + c = 0 dentro de [0,W] x [ytop,ybot] (coordenadas do quadro)."""
    a, b, c = l
    pts = []
    for x in (0, W):
        if abs(b) > 1e-9:
            y = -(a * x + c) / b
            if ytop <= y <= ybot:
                pts.append((x, y))
    for y in (ytop, ybot):
        if abs(a) > 1e-9:
            x = -(b * y + c) / a
            if 0 <= x <= W:
                pts.append((x, y))
    return np.array(pts[:2]) if len(pts) >= 2 else None


plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
FIG_W = 6.67
panel_h = (2 * H + GAP) / W * (FIG_W - 0.08) / 2
fig, axs = plt.subplots(1, 2, figsize=(FIG_W, panel_h + 0.04), dpi=300)
plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.024)
for ax in axs:
    ax.imshow(stack(*rgb), interpolation="lanczos")
    ax.set_xlim(0, W)
    ax.set_ylim(2 * H + GAP, 0)
    ax.axis("off")
    for nome, yy in ((A, 0), (B, H + GAP)):
        ax.text(14, yy + 14, nome, va="top", ha="left", fontsize=7, color="white",
                bbox=dict(boxstyle="square,pad=0.2", fc="black", ec="none", alpha=0.7))
    ax.text(W - 14, 14, "(a)" if ax is axs[0] else "(b)", va="top", ha="right", fontsize=8, fontweight="bold",
            color="black", bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85))

# (a) correspondencias
ax = axs[0]
segs_out = np.stack([to_a(pa[idx_out_s]), to_b(pb[idx_out_s])], axis=1)
segs_in = np.stack([to_a(pa[idx_in_s]), to_b(pb[idx_in_s])], axis=1)
ax.add_collection(LineCollection(segs_out, colors="#ff2a2a", linewidths=0.35, alpha=0.6))
ax.add_collection(LineCollection(segs_in, colors="#22ee44", linewidths=0.55, alpha=0.95))
for p, c in ((np.r_[to_a(pa[idx_in_s]), to_b(pb[idx_in_s])], "#22ee44"),
             (np.r_[to_a(pa[idx_out_s]), to_b(pb[idx_out_s])], "#ff2a2a")):
    ax.scatter(p[:, 0], p[:, 1], s=1.6, c=c, linewidths=0, zorder=3)
n_in, n_out = int(inl.sum()), int((~inl).sum())
ax.text(0.56 * W, 2 * H + GAP - 14, f"inliers {n_in} / outliers {n_out}", va="bottom", ha="center", fontsize=7,
        linespacing=1.15, color="white", bbox=dict(boxstyle="square,pad=0.25", fc="black", ec="none", alpha=0.7))

# (b) geometria epipolar: ponto em A -> reta l' = F x em B; ponto em B -> reta l = F^T x' em A
ax = axs[1]
for k, i in enumerate(sel):
    c = cols[k]
    xa = np.array([*pa[i], 1.0])
    xb = np.array([*pb[i], 1.0])
    for line, off, pt, pt_off in ((F @ xa, H + GAP, pb[i], H + GAP),   # reta em B
                                  (F.T @ xb, 0, pa[i], 0)):            # reta em A
        seg = clip_line(line, y0, y1 - 1)
        if seg is not None:
            ax.plot(seg[:, 0], seg[:, 1] - y0 + off, color=c, lw=0.7, alpha=0.95)
    for p, off in ((pa[i], 0), (pb[i], H + GAP)):
        ax.scatter(p[0], p[1] - y0 + off, s=9, color=c, edgecolors="white", linewidths=0.4, zorder=4)

fig.savefig(OUT, format="jpg", dpi=300, pil_kwargs={"quality": 85})
print("figura", OUT, f"{FIG_W:.2f} x {panel_h + 0.04:.2f} pol")
