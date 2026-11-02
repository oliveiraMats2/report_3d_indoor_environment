"""G2: nuvem esparsa do COLMAP (cozinha) em scatter 3D, cor = altura sobre o chao (cm).
Fontes: cozinha/results/config_cozinha/{scale/points3d_cm.npy, scale/centers_cm.npy, planes/floor.npy}
Rodar: /home/moliveira/Documents/3d_indoor_environment/.venv/bin/python G2_nuvem_esparsa_viridis.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha/"
OUT = "/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/G2_nuvem_esparsa_viridis.png"
AZIM, ELEV = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (30, 35)
N_DRAW, HMAX, HMIN = 120000, 270.0, -10.0
BG, FG = "#0b0e20", "white"

p = np.load(R + "scale/points3d_cm.npy"); cen = np.load(R + "scale/centers_cm.npy")
f = np.load(R + "planes/floor.npy"); n, d = f[:3], f[3]
h = p @ n + d; hc = cen @ n + d
# base horizontal ortonormal, alinhada aos eixos principais da nuvem recortada
keep = (h >= HMIN) & (h <= HMAX)
e1 = np.cross(n, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
q = p[keep] - p[keep].mean(0); xy = np.c_[q @ e1, q @ e2]
w, V = np.linalg.eigh(np.cov(xy.T)); V = V[:, ::-1]
b1 = V[0, 0] * e1 + V[1, 0] * e2; b2 = V[0, 1] * e1 + V[1, 1] * e2
o = p[keep].mean(0)
X = (p - o) @ b1; Y = (p - o) @ b2
Xc = (cen - o) @ b1; Yc = (cen - o) @ b2
# recorta horizontalmente (1o..99o percentil) para ignorar espalhados da janela
lo1, hi1 = np.percentile(X[keep], [0.5, 99.5]); lo2, hi2 = np.percentile(Y[keep], [0.5, 99.5])
keep &= (X >= lo1) & (X <= hi1) & (Y >= lo2) & (Y <= hi2)
idx = np.flatnonzero(keep)
rng = np.random.default_rng(0)
sel = rng.choice(idx, min(N_DRAW, idx.size), replace=False)

fig = plt.figure(figsize=(11, 8.2), facecolor=BG)
ax = fig.add_subplot(111, projection="3d", facecolor=BG)
sc = ax.scatter(X[sel], Y[sel], h[sel], c=h[sel], cmap="viridis", s=0.5, alpha=0.7,
                vmin=0, vmax=HMAX, linewidths=0, depthshade=False)
ax.scatter(Xc, Yc, hc, c="white", s=1.2, alpha=0.9, linewidths=0, label="centros das câmeras (2326)")
ax.view_init(elev=ELEV, azim=AZIM)
ax.set_box_aspect((hi1 - lo1, hi2 - lo2, HMAX - HMIN))
ax.set_xlim(lo1, hi1); ax.set_ylim(lo2, hi2); ax.set_zlim(HMIN, HMAX)
for a, t in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel),
                ("X horizontal (cm)", "Y horizontal (cm)", "altura sobre o chão (cm)")):
    a(t, color=FG, labelpad=8)
ax.tick_params(colors=FG, labelsize=8)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((0.04, 0.05, 0.13, 1)); a._axinfo["grid"]["color"] = (0.75, 0.75, 0.8, 0.5)
cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02); cb.set_label("altura sobre o chão (cm)", color=FG)
cb.ax.yaxis.set_tick_params(color=FG); plt.setp(cb.ax.get_yticklabels(), color=FG)
leg = ax.legend(loc="upper left", facecolor=BG, edgecolor="none", labelcolor=FG, markerscale=6, fontsize=9)
ax.set_title("Nuvem esparsa do COLMAP (SfM), cozinha\n727.716 pontos no total  |  " + f"{sel.size:,}".replace(",", ".") + " desenhados (amostra aleatória)",
             color=FG, fontsize=13)
fig.text(0.01, 0.015, "Cor = altura sobre o chão (plano do chão ajustado). Recorte: altura de -10 a 270 cm (teto removido) e 1% de extremos horizontais; sobram 704.596 pontos, dos quais se desenha uma amostra.\n"
         "Fonte: cozinha/results/config_cozinha/scale/points3d_cm.npy, centers_cm.npy, planes/floor.npy", color="#9aa0b8", fontsize=8)
fig.savefig(OUT, dpi=220, facecolor=BG, bbox_inches="tight")
print("ok", OUT, "desenhados", sel.size, "de", int(keep.sum()), "recortados; total", len(p))
