"""Gera esparsa_topo.jpg (nuvem esparsa vista de cima, sem teto, com trajetoria) e planta.jpg (planta baixa).
Uso: R/.venv/bin/python planta.py   (somente leitura em REPO; saidas em images/report/)."""
import numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from pathlib import Path

REPO = Path("/home/moliveira/Documents/3d_indoor_environment/cozinha")
RUN = REPO / "results/config_cozinha"
OUT = Path(__file__).resolve().parents[1]
H_MAX = 270.0          # cm acima do chao: remove o teto (~289,6 cm)
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})

T = np.load(RUN / "planes/T_floor.npy")                 # COLMAP(cm) -> referencial do chao (z = altura)
fl = lambda p: p @ T[:3, :3].T + T[:3, 3]
pts = np.load(RUN / "clean/sparse/points.npy")           # 286458 pontos, cm
col = np.load(RUN / "clean/sparse/colors.npy")
lab = np.load(RUN / "planes/labels.npy")                 # rotulos (>=2: parede)
P = fl(pts)
C = fl(np.load(RUN / "scale/centers_cm.npy"))            # 2326 centros, cm
n = len(C)

# retangulo das paredes: mesmo procedimento de src/cloud/floorplan.py (minAreaRect + mediana, percentis 1-99, banda 5 cm)
w = P[lab >= 2, :2].astype(np.float32)
(_, _), (_, _), ang = cv2.minAreaRect(w)
a = np.deg2rad(ang); rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
loc = w @ rot.T
lo, hi = np.percentile(loc, [1, 99], axis=0)
for k in range(2):
    for e in (lo, hi):
        nr = loc[np.abs(loc[:, k] - e[k]) <= 5.0, k]
        e[k] = np.median(nr)
(x0, y0), (x1, y1) = lo, hi
W, L = x1 - x0, y1 - y0                                   # 274,6 x 444,0

# contorno do app Medida (273 x 429), ajustado pelo room_fill
poly = np.load(RUN / "mesh/room_fill/footprint_cm.npy").reshape(-1, 2)
Tr = np.load(RUN / "mesh/room_fill/T_room.npy")
tren = fl((np.c_[poly, np.zeros(len(poly)), np.ones(len(poly))] @ Tr.T)[:, :3])[:, :2]

# convencao de desenho: horizontal = comprimento (y do chao), vertical = largura (-x do chao)
H = lambda xy: np.c_[xy[:, 1], -xy[:, 0]]
M = 25
xl, xr = y0 - M, y1 + M
yb, yt = -x1 - M, -x0 + M

keep = (P[:, 2] > 5) & (P[:, 2] < H_MAX)
Q = H(P[keep, :2]); cq = col[keep]
Cs = H(C[:, :2])

def base(ax):
    ax.set_aspect("equal"); ax.set_xlim(xl, xr); ax.set_ylim(yb, yt)
    ax.set_xlabel("comprimento (cm)"); ax.set_ylabel("largura (cm)")
    ax.grid(True, lw=0.3, alpha=0.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---------- (a) esparsa + trajetoria ----------
fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=170)
ax.scatter(Q[:, 0], Q[:, 1], s=0.25, c=np.clip(cq, 0, 1) if cq.max() <= 1 else cq / 255, marker=".", linewidths=0, rasterized=True)
ax.add_patch(Rectangle((y0, -x1), L, W, fill=False, ec="0.35", lw=0.8, ls=":"))
ax.plot(Cs[:, 0], Cs[:, 1], color="0.2", lw=0.3, alpha=0.5, zorder=3)
sc = ax.scatter(Cs[:, 0], Cs[:, 1], c=np.arange(n), cmap="plasma", s=3, zorder=4)
ax.scatter(*Cs[0], s=70, marker="o", fc="#00c000", ec="k", zorder=6, label="início (quadro 0)")
ax.scatter(*Cs[-1], s=90, marker="s", fc="#e00000", ec="k", zorder=6, label=f"fim (quadro {n - 1})")
base(ax)
cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02); cb.set_label("índice do quadro")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=10)
fig.savefig(OUT / "esparsa_topo.jpg", format="jpg", pil_kwargs={"quality": 82}, bbox_inches="tight")
plt.close(fig)

# ---------- (b) planta baixa ----------
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=170)
Hh = Q
ax.hist2d(Hh[:, 0], Hh[:, 1], bins=[np.arange(xl, xr + 5, 5), np.arange(yb, yt + 5, 5)], cmap="Greys", norm=matplotlib.colors.PowerNorm(0.4), cmin=1)
ax.add_patch(Rectangle((y0, -x1), L, W, fill=False, ec="#c00000", lw=1.6, label=f"modelo: {W:.1f} × {L:.1f} cm".replace(".", ",")))
t = H(tren); ax.add_patch(Polygon(t, closed=True, fill=False, ec="#0060c0", lw=1.4, ls="--", label="app Medida: 273 × 429 cm"))
ax.plot(Cs[:, 0], Cs[:, 1], color="#e08000", lw=0.4, alpha=0.7, label="trajetória da câmera")
base(ax)
kw = dict(arrowstyle="<->", color="#c00000", lw=1)
ax.annotate("", (y0, yb + 8), (y1, yb + 8), arrowprops=kw)
ax.text((y0 + y1) / 2, yb + 11, f"{L:.1f} cm (modelo)".replace(".", ","), color="#c00000", ha="center", fontsize=9.5, bbox=dict(fc="w", ec="none", pad=1))
ax.annotate("", (y1 - 12, -x1), (y1 - 12, -x0), arrowprops=kw)
ax.text(y1 - 16, -(x0 + x1) / 2, f"{W:.1f} cm (modelo)".replace(".", ","), color="#c00000", ha="right", va="center", fontsize=9.5, rotation=90, bbox=dict(fc="w", ec="none", pad=1))
ax.text(y0 + L * 0.5, -x0 - 12, "app Medida: 273 × 429 cm", color="#0060c0", ha="center", fontsize=9.5, bbox=dict(fc="w", ec="none", pad=1))
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9.5)
fig.savefig(OUT / "planta.jpg", format="jpg", pil_kwargs={"quality": 82}, bbox_inches="tight")
plt.close(fig)

print(f"retangulo {W:.2f} x {L:.2f} cm; pontos esparsos 5<h<{H_MAX:.0f}: {keep.sum()} de {len(P)}; cameras {n}")
print("altura camera mediana", np.median(C[:, 2]), "contorno do app Medida", np.ptp(tren, axis=0))
