"""G10: remocao de outliers em 2D (nuvem densa fatiada + histogramas). Somente leitura dos dados.
Uso: R/.venv/bin/python G10_remocao_outliers_2d.py"""
import re, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
R = "/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha/"
OUT = "/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/G10_remocao_outliers_2d.png"
M, HP, HT, K, KS = 20.0, -10.0, 278.0, 20, 2.0
Z0, Z1 = 100.0, 110.0
br = lambda t: re.sub(r"(?<=\d)\.(?=\d)", ",", re.sub(r"(?<=\d),(?=\d)", "\x00", t)).replace("\x00", ".")
br0 = br
br = lambda t: re.sub(r"(?<=\d) x (?=\d)", " × ", br0(t))
def dist_poly(xy, fp):
    dentro = MPath(np.vstack([fp, fp[:1]])).contains_points(xy); d = np.full(len(xy), np.inf)
    for a, b in zip(fp, np.roll(fp, -1, 0)):
        ab = b - a; t = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(xy - (a + t[:, None] * ab), axis=1))
    return np.where(dentro, 0.0, d)
# nuvem densa no referencial da sala (mesma regra de gera_malha_limpa.py)
p = np.load(R+"clean/patch_match/points.npy").astype(np.float64).reshape(-1, 3)
Ti = np.linalg.inv(np.load(R+"mesh/room_fill/T_room.npy")); fp = np.load(R+"mesh/room_fill/footprint_cm.npy")
q = p @ Ti[:3, :3].T + Ti[:3, 3]
c_fora = dist_poly(q[:, :2], fp) > M
c_alt = ((q[:, 2] < HP) | (q[:, 2] > HT)) & ~c_fora
base = ~(c_fora | (q[:, 2] < HP) | (q[:, 2] > HT))
dk = cKDTree(q).query(q, k=K+1, workers=-1)[0][:, 1:].mean(1)
lim = dk[base].mean() + KS*dk[base].std(); c_fl = base & (dk > lim)
cat = np.zeros(len(q), int); cat[c_fl] = 3; cat[c_alt] = 2; cat[c_fora] = 1   # 0 mantido
n = [(cat == i).sum() for i in range(4)]
print("dense", len(q), n, lim)
# nuvem esparsa
sp = np.load(R+"sfm/points3d.npy"); mk = np.load(R+"report_filtragem/outlier_mask_std2.0.npy")
dks = cKDTree(sp).query(sp, k=K+1, workers=-1)[0][:, 1:].mean(1)
lims = dks.mean() + KS*dks.std()
print("sparse", len(sp), mk.sum(), lims, (dks > lims).sum())
cor = {0: "#2ca02c", 1: "#d62728", 2: "#1f77b4", 3: "#ff7f0e"}
nome = {1: "fora do contorno > 20 cm", 2: "altura fora de [-10, 278] cm", 3: "flutuante (dist. média > limiar)"}
_st = matplotlib.axes.Axes.set_title
matplotlib.axes.Axes.set_title = lambda self, t, *a, **k: _st(self, br(t), *a, **k)
_tx = matplotlib.axes.Axes.text
matplotlib.axes.Axes.text = lambda self, x, y, t, *a, **k: _tx(self, x, y, br(t), *a, **k)
_ft = matplotlib.figure.Figure.suptitle
_ftx = matplotlib.figure.Figure.text
matplotlib.figure.Figure.text = lambda self, x, y, t, *a, **k: _ftx(self, x, y, br(t), *a, **k)
fig = plt.figure(figsize=(16, 12), dpi=100)
gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)
# (a) fatia
sl = (q[:, 2] >= Z0) & (q[:, 2] < Z1)
xl, yl = (-260, 330), (-330, 430)
def plano(ax, mostra_rem=True, titulo=""):
    for i in ([1, 3, 0] if mostra_rem else [0]):
        s = sl & (cat == i)
        if i == 0 and False: pass
        ax.scatter(q[s, 0], q[s, 1], s=0.4, c=cor[i], lw=0, rasterized=True)
    ax.add_patch(Polygon(fp, closed=True, fill=False, ec="k", lw=1.8))
    ax.set_aspect("equal"); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xlabel("x da sala (cm)"); ax.set_ylabel("y da sala (cm)")
ax = fig.add_subplot(gs[0, 0]); plano(ax)
# faixa de 20 cm: polígono expandido (buffer) via shapely se houver, senão contorno por distância
gx, gy = np.meshgrid(np.linspace(*xl, 400), np.linspace(*yl, 600))
dd = dist_poly(np.c_[gx.ravel(), gy.ravel()], fp).reshape(gx.shape)
ax.contour(gx, gy, dd, levels=[M], colors="k", linestyles="--", linewidths=1.4)
ax.contourf(gx, gy, dd, levels=[1e-6, M], colors=["#bbbbbb"], alpha=0.35)
ax.set_title(f"(a) Fatia {Z0:.0f} a {Z1:.0f} cm, vista de cima ({sl.sum():,} pontos)", fontsize=12, loc="left")
nf = lambda i: (sl & (cat == i)).sum()
ax.legend(handles=[Patch(color=cor[0], label=br(f"mantido ({nf(0):,})")),
    Patch(color=cor[1], label=br(f"fora do contorno + 20 cm ({nf(1):,})")),
    Patch(color=cor[3], label=br(f"flutuante, dist. média > {lim:.3f} cm ({nf(3):,})")),
    Line2D([], [], color="k", lw=1.8, label="contorno da sala (274,6 x 444,0 cm)"),
    Line2D([], [], color="k", ls="--", lw=1.4, label="contorno + 20 cm (faixa cinza)")],
    loc="upper right", fontsize=8, markerscale=2, framealpha=0.95)
# (b) histogramas
gb = gs[0, 1].subgridspec(2, 1, hspace=0.75)
for k, (d, ref, tit, lm, nrem) in enumerate([(dks, None, "nuvem esparsa (SfM)", lims, int(mk.sum())), (dk[base], None, "nuvem densa (pontos dentro do contorno e da altura)", lim, int(c_fl.sum()))]):
    a = fig.add_subplot(gb[k]); hi = np.percentile(d, 99.8)
    b = np.linspace(0, hi, 120); ok = (~mk) if k == 0 else (d <= lm)
    a.hist(d[ok], bins=b, color=cor[0], label="mantidos"); a.hist(d[~ok], bins=b, color=cor[1], label="removidos")
    a.axvline(lm, color="k", ls="--", lw=1.6); a.set_yscale("log")
    a.set_xlabel("distância média aos 20 vizinhos mais próximos (" + ("unid. do modelo SfM" if k == 0 else "cm") + ")"); a.set_ylabel("nº de pontos (log)")
    a.set_title(f"{tit}: {len(d):,} pontos, {nrem:,} removidos ({100*nrem/len(d):.2f}%)", fontsize=10, loc="left", pad=8)
    a.text(lm, a.get_ylim()[1]*0.6, f"  limiar = média + 2 desvios\n  = {d.mean():.3f} + 2 x {d.std():.3f} = {lm:.3f}", fontsize=9, va="top")
    if k == 0: a.legend(loc="center right", fontsize=9)
    if k == 0: a.text(0, 1.38, "(b) Regra do flutuante: remove se a distância média aos 20 vizinhos > média + 2 desvios", transform=a.transAxes, fontsize=12)
# (c) perfil lateral
ax = fig.add_subplot(gs[1, 0])
yb = (np.abs(q[:, 0] - (-10)) < 5)  # fatia fina em x (10 cm) no meio da sala
for i in [1, 2, 3, 0]:
    s = yb & (cat == i); ax.scatter(q[s, 1], q[s, 2], s=0.6, c=cor[i], lw=0, rasterized=True)
ymin, ymax = fp[:, 1].min(), fp[:, 1].max()
ax.axhline(HT, color=cor[2], lw=1.8); ax.axhline(HP, color=cor[2], lw=1.8)
ax.axhspan(HP, HT, color=cor[0], alpha=0.07)
ax.axvline(ymin - 0, color="k", lw=1.2); ax.axvline(ymax, color="k", lw=1.2)
ax.axvspan(ymin - M, ymin, color="#bbbbbb", alpha=0.4); ax.axvspan(ymax, ymax + M, color="#bbbbbb", alpha=0.4)
ax.text(-100, HT + 8, "limite do teto: h = 278 cm (acima, removido)", color=cor[2], fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
ax.text(-100, HP - 24, "limite do chão: h = -10 cm (abaixo, removido)", color=cor[2], fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
ax.set_xlim(ymin - 80, ymax + 80); ax.set_ylim(-80, 400); ax.set_xlabel("y da sala (cm)"); ax.set_ylabel("altura h sobre o piso (cm)")
ax.set_title(f"(c) Corte vertical (faixa de 10 cm em x = -10 cm): {yb.sum():,} pontos", fontsize=12, loc="left")
ax.legend(handles=[Patch(color=cor[0], label="mantido"), Patch(color=cor[1], label=nome[1]), Patch(color=cor[2], label=nome[2]),
    Patch(color=cor[3], label=nome[3])], loc="upper right", fontsize=8.5, framealpha=0.95)
# (d) antes -> depois
gd = gs[1, 1].subgridspec(1, 2, wspace=0.05)
a1 = fig.add_subplot(gd[0]); a2 = fig.add_subplot(gd[1])
for a, ti in [(a1, "antes"), (a2, "depois")]:
    for i in ([0, 1, 3] if ti == "antes" else [0]):
        s = sl & (cat == i); a.scatter(q[s, 0], q[s, 1], s=0.4, c=("#555555" if ti == "antes" else cor[0]), lw=0, rasterized=True)
    a.add_patch(Polygon(fp, closed=True, fill=False, ec="k", lw=1.2)); a.set_aspect("equal"); a.set_xlim(*xl); a.set_ylim(*yl); a.set_xlabel("x (cm)")
a1.set_ylabel("y (cm)"); a2.set_yticklabels([])
a1.set_title(f"antes: {sl.sum():,}", fontsize=11); a2.set_title(f"depois: {nf(0):,}", fontsize=11)
a1.text(0.5, 1.12, f"(d) Fatia {Z0:.0f}-{Z1:.0f} cm: antes e depois da limpeza", transform=a1.transAxes, fontsize=12, ha="left", x=0) if False else None
fig.text(0.535, 0.485, f"(d) Fatia {Z0:.0f} a {Z1:.0f} cm antes e depois ({nf(1)+nf(3):,} removidos, {100*(nf(1)+nf(3))/sl.sum():.1f}%)", fontsize=12)
tot = n[1]+n[2]+n[3]
fig.suptitle(f"Remoção de outliers da nuvem densa da cozinha: {len(q):,} pontos, {tot:,} removidos\n({n[1]:,} fora do contorno + {n[2]:,} fora da faixa de altura + {n[3]:,} flutuantes), {len(q)-tot:,} mantidos", fontsize=13, y=1.0)
from matplotlib.ticker import FuncFormatter, ScalarFormatter
def _fmt(v, pos):
    t = ScalarFormatter(useMathText=False).format_data_short(v) if False else ("%g" % v)
    return t.replace("-", "-").replace(".", ",")
for a in fig.axes:
    a.xaxis.set_major_formatter(FuncFormatter(_fmt))
    if a.get_yscale() != "log": a.yaxis.set_major_formatter(FuncFormatter(_fmt))
a2.tick_params(labelleft=False)
fig.savefig(OUT, dpi=220, bbox_inches="tight"); print("ok")
