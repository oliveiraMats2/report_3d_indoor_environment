"""E3: triangulacao com ruido, duas baselines. Dados: modelo COLMAP da cozinha."""
import numpy as np, cv2, pycolmap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
R0 = "/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha/"
rec = pycolmap.Reconstruction(R0 + "exports/colmap/sparse/0")
S = 29.3745  # cm por unidade
pe = np.load(R0 + "sfm/point_error.npy")
cam = list(rec.cameras.values())[0]
fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]); D = np.array([k1, k2, p1, p2])
by = {i.name: i for i in rec.images.values()}
base = 1150
ks = list(range(1, 21))
names = [f"img_{base+k:04d}.png" for k in [0] + ks]
common = None
for n in names:
    s = {p.point3D_id for p in by[n].points2D if p.has_point3D()}
    common = s if common is None else common & s
# ponto com erro proximo do medio e profundidade mediana
im0 = by[names[0]]
def depth(pid): return (im0.cam_from_world() * rec.points3D[pid].xyz)[2]
cand = [i for i in common if rec.points3D[i].track.length() >= 8]
print("candidatos", len(common), len(cand))
dd = np.array([depth(i) * S for i in cand]); ee = np.array([rec.points3D[i].error for i in cand])
med = np.median(dd)
sc = np.abs(ee - 0.565) / 0.565 + np.abs(dd - med) / med
pid = cand[int(np.argmin(sc))]
X = rec.points3D[pid].xyz
print("ponto", pid, "erro", rec.points3D[pid].error, "prof cm", depth(pid) * S)

def obs(name):
    im = by[name]
    for p in im.points2D:
        if p.has_point3D() and p.point3D_id == pid: return np.array(p.xy)
def Pn(name):
    T = by[name].cam_from_world().matrix()  # 3x4
    return T
def tri(n0, n1, sigma, N, rng):
    x0, x1 = obs(n0), obs(n1)
    a = x0 + rng.normal(0, sigma, (N, 2)); b = x1 + rng.normal(0, sigma, (N, 2))
    u0 = cv2.undistortPoints(a.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    u1 = cv2.undistortPoints(b.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    P0 = Pn(n0); P1 = Pn(n1)
    Xh = cv2.triangulatePoints(P0, P1, u0.T, u1.T)
    Xw = (Xh[:3] / Xh[3]).T
    # coordenadas na camera 0: x lateral, z profundidade (cm)
    Xc = (P0[:, :3] @ Xw.T + P0[:, 3:4]).T * S
    return Xc
SIG = 0.565 / np.sqrt(np.pi / 2)  # por eixo, para residuo medio |e| = 0,565 px
N = 500
rng = np.random.default_rng(0)
def base_cm(n0, n1):
    return np.linalg.norm(by[n0].projection_center() - by[n1].projection_center()) * S
pairs = [("img_1150.png", "img_1151.png"), ("img_1150.png", "img_1154.png")]
res = []
for p in pairs:
    Xc = tri(*p, SIG, N, rng); res.append((p, base_cm(*p), Xc))
    print(p, "baseline cm %.2f" % res[-1][1], "media", Xc.mean(0), "std x,y,z cm", Xc.std(0))
# curva
bl, sz, sx = [], [], []
for k in ks:
    n1 = f"img_{base+k:04d}.png"
    Xc = tri(names[0], n1, SIG, 500, np.random.default_rng(k))
    bl.append(base_cm(names[0], n1)); sz.append(Xc[:, 2].std()); sx.append(Xc[:, 0].std())
bl, sz = np.array(bl), np.array(sz)
z = depth(pid) * S; f = (fx + fy) / 2
theo = z**2 * SIG * np.sqrt(2) / (f * bl)
print("curva", np.c_[bl, sz, theo].round(2).tolist())
# figura
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(15, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.38, wspace=0.28)
col = ["#1f77b4", "#d95f02"]
zm = res[0][2][:, 2].mean()
for j, (p, b, Xc) in enumerate(res):
    ax = fig.add_subplot(gs[0, j])
    ax.scatter(Xc[:, 0], Xc[:, 2], s=7, alpha=.45, color=col[j], label=f"{N} triangulações")
    m = Xc[:, [0, 2]].mean(0); C = np.cov(Xc[:, [0, 2]].T)
    w, v = np.linalg.eigh(C); ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(m, 4 * np.sqrt(w[1]), 4 * np.sqrt(w[0]), angle=ang, fill=False, ec="k", lw=1.6, label="elipse 2σ"))
    ax.plot(*m, "k+", ms=10)
    # camera 0 e 1 em coordenadas da camera 0
    c1 = (by[p[0]].cam_from_world() * by[p[1]].projection_center()) * S
    ax.set_title(f"{p[0][:-4]} e {p[1][:-4]}: baseline {b:.1f} cm\nσ profundidade = {Xc[:,2].std():.2f} cm, σ lateral = {Xc[:,0].std():.2f} cm", fontsize=11)
    ax.set_xlabel("x lateral (cm, câmera da img_1150)"); ax.set_ylabel("z profundidade (cm)")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=.3)
    ax.text(.98, .03, f"profundidade média {m[1]:.0f} cm", transform=ax.transAxes, ha="right", fontsize=9)
# painel: duas escalas iguais para comparar
ax = fig.add_subplot(gs[0, 2])
for j, (p, b, Xc) in enumerate(res):
    ax.scatter(Xc[:, 0] - Xc[:, 0].mean(), Xc[:, 2] - Xc[:, 2].mean(), s=7, alpha=.4, color=col[j], label=f"baseline {b:.1f} cm")
ax.set_title("Mesma escala, centradas na média\n(nuvem da baseline maior bem mais compacta)", fontsize=11)
ax.set_xlabel("desvio lateral (cm)"); ax.set_ylabel("desvio em profundidade (cm)")
ax.legend(fontsize=9); ax.grid(alpha=.3)
ax = fig.add_subplot(gs[1, :2])
ax.plot(bl, sz, "o-", color="#1b7837", label="simulação (500 amostras por baseline)")
ax.plot(bl, theo, "--", color="gray", label=r"aproximação $\sigma_z\approx z^2\,\sqrt{2}\,\sigma/(f\,b)$")
for (p, b, Xc), c in zip(res, col): ax.plot(b, Xc[:, 2].std(), "s", ms=11, mfc="none", mec=c, mew=2)
ax.set_xlabel("baseline entre as câmeras (cm)"); ax.set_ylabel("σ da profundidade (cm)")
ax.set_title(f"Incerteza em profundidade × baseline (ponto a {z:.0f} cm; pares img_1150 e img_1151 a img_1170)", fontsize=11)
ax.legend(); ax.grid(alpha=.3)
ax = fig.add_subplot(gs[1, 2])
ax.hist(pe, bins=np.linspace(0, 2, 81), color="#6a51a3", alpha=.85)
ax.axvline(pe.mean(), color="k", ls="--", label=f"média 0,565 px")
ax.set_xlabel("erro de reprojeção por ponto (px)"); ax.set_ylabel("pontos 3D")
ax.set_title(f"Erro de reprojeção real ({len(pe):,} pontos)".replace(",", " "), fontsize=11); ax.legend()
fig.suptitle(f"Triangulação com ruído nos keypoints (σ = 0,451 px por eixo, resíduo médio 0,565 px), ponto 3D {pid} da cozinha", fontsize=13)
import re
from matplotlib.ticker import FuncFormatter, ScalarFormatter
def _v(x, pos=None):
    return ScalarFormatter().format_data_short(x) if False else (f"{x:.10g}").replace(".", ",")
for a in fig.axes:
    for ax_ in (a.xaxis, a.yaxis):
        ax_.set_major_formatter(FuncFormatter(_v))
for t in fig.findobj(matplotlib.text.Text):
    t.set_text(re.sub(r"(\d)\.(\d)", r"\1,\2", t.get_text()))
fig.savefig("E3_triangulacao_ruido.png", dpi=220, bbox_inches="tight")
