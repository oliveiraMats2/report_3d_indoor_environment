"""E2: ESQUEMA didatico (fora de escala) de geometria epipolar com dados reais.
Cozinha, par img_0771 / img_0789 (modelo COLMAP). Posicoes de desenho livres; miniaturas, pixels,
epipolos (px) e linha de base (cm) sao reais."""
import os, numpy as np, cv2, pycolmap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

R0 = "/home/moliveira/Documents/3d_indoor_environment/cozinha"
MODEL = R0 + "/results/config_cozinha/exports/colmap/sparse/0"
FR = R0 + "/frames/all_stride10/"
OUT = "/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/E2_geometria_epipolar.png"
S = 29.3745
NA, NB = os.environ.get("NA", "img_0771"), os.environ.get("NB", "img_0789")
def fm(v, d=1): return f"{v:.{d}f}".replace(".", ",")

rec = pycolmap.Reconstruction(MODEL)
ims = {i.name: i for i in rec.images.values()}
ia, ib = ims[NA + ".png"], ims[NB + ".png"]
cam = rec.cameras[ia.camera_id]
fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]); dist = np.array([k1, k2, p1, p2])
W, H = cam.width, cam.height
def pose(i):
    T = i.cam_from_world(); return np.array(T.rotation.matrix()), np.array(T.translation)
Ra, ta = pose(ia); Rb, tb = pose(ib)
Ca, Cb = -Ra.T @ ta * S, -Rb.T @ tb * S
base = np.linalg.norm(Ca - Cb)
pa = {p.point3D_id: p.xy for p in ia.points2D if p.has_point3D()}
pb = {p.point3D_id: p.xy for p in ib.points2D if p.has_point3D()}
common = set(pa) & set(pb)
best = None
for k in common:
    P = rec.points3D[k]; ua, ub = pa[k], pb[k]
    if not (700 < ua[0] < 1300 and 400 < ua[1] < 700 and 700 < ub[0] < 1300 and 400 < ub[1] < 700): continue
    if P.track.length() < 40 or P.error > 1.0: continue
    za = (Ra @ P.xyz + ta)[2] * S
    if not 60 < za < 140: continue
    if best is None or P.track.length() > best[0]: best = (P.track.length(), k)
kid = best[1]; P3 = rec.points3D[kid]; X = P3.xyz * S
def undist(u): return cv2.undistortPoints(np.array([[u]], np.float64), K, dist, P=K)[0, 0]
x1, x2 = undist(pa[kid]), undist(pb[kid])
Rrel = Rb @ Ra.T; trel = tb - Rrel @ ta
tx = np.array([[0, -trel[2], trel[1]], [trel[2], 0, -trel[0]], [-trel[1], trel[0], 0]])
Ki = np.linalg.inv(K); F = Ki.T @ tx @ Rrel @ Ki
U, Sg, Vt = np.linalg.svd(F); e1h = Vt[-1] / Vt[-1][2]; e2h = U[:, -1] / U[:, -1][2]
l2 = F @ np.array([x1[0], x1[1], 1]); l1 = F.T @ np.array([x2[0], x2[1], 1])
res = abs(l2 @ np.array([x2[0], x2[1], 1])) / np.hypot(*l2[:2])
z1 = (Ra @ P3.xyz + ta)[2] * S; z2 = (Rb @ P3.xyz + tb)[2] * S
print("par", NA, NB, "comuns", len(common), "base cm", base)
print("ponto", kid, "trilha", P3.track.length(), "erro", P3.error, "prof", z1, z2)
print("x1", x1, "x2", x2, "resid", res, "e1", e1h[:2], "e2", e2h[:2])

# ---------- miniaturas (recorte em volta do keypoint) ----------
CWc, CHc = 560, 340
def crop(name, x):
    im = cv2.undistort(cv2.imread(FR + name + ".png"), K, dist); im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    u0 = int(np.clip(x[0] - CWc / 2, 0, W - CWc)); v0 = int(np.clip(x[1] - CHc / 2, 0, H - CHc))
    return im[v0:v0 + CHc, u0:u0 + CWc], np.array([x[0] - u0, x[1] - v0]), (u0, v0)
th1, xp1, o1 = crop(NA, x1); th2, xp2, o2 = crop(NB, x2)

# ---------- esquema (unidades livres; 100 px por unidade) ----------
GW, GH = 26.0, 16.0
C1 = np.array([3.0, 3.6]); C2 = np.array([23.0, 3.6]); Xd = np.array([13.0, 14.6])
KS = 5.0 / CWc                      # unidades por pixel de miniatura
TILT = np.radians(float(os.environ.get("TILT", 33)))
def make_map(sign, xp, target):
    a = np.array([KS, sign * KS * np.tan(TILT)]); b = np.array([0.0, -KS])   # eixos u e v (v para baixo)
    O = target - xp[0] * a - xp[1] * b
    return lambda q: O + q[0] * a + q[1] * b, a, b
t1 = float(os.environ.get("T1", 0.32)); t2 = float(os.environ.get("T2", 0.32))
tg1 = C1 + t1 * (Xd - C1); tg2 = C2 + t2 * (Xd - C2)
M1, a1, b1 = make_map(-1, xp1, tg1)   # plano 1: inclina descendo para a direita
M2, a2, b2 = make_map(+1, xp2, tg2)   # plano 2: espelhado

def line_dir(l, a, b):
    d = np.array([l[1], -l[0]]); d = d / np.linalg.norm(d)          # direcao da reta em pixels
    return d[0] * a + d[1] * b, d
def hit_baseline(pt, dv):
    s = (C1[1] - pt[1]) / dv[1]; return pt + s * dv, s
quads, epi = [], []
for M, a, b, xp, l, sgn, th in [(M1, a1, b1, xp1, l1, +1, th1), (M2, a2, b2, xp2, l2, -1, th2)]:
    dv, d = line_dir(l, a, b)
    pt = M(xp)
    if dv[0] * sgn < 0: dv = -dv                                      # aponta para o outro centro
    dv = dv / np.linalg.norm(dv); e, s = hit_baseline(pt, dv); epi.append((pt, dv, e))
E1d, E2d = epi[0][2], epi[1][2]
print("epipolos no esquema", E1d, E2d)

fig = plt.figure(figsize=(GW / 2, GH / 2), dpi=200)       # 13 x 8 pol
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, GW); ax.set_ylim(0, GH); ax.axis("off")
ax.add_patch(Polygon([C1, C2, Xd], closed=True, fc="#d6f0c0", ec="none", zorder=1))
def quadpts(M): return [M((0, 0)), M((CWc, 0)), M((CWc, CHc)), M((0, CHc))]
CVW, CVH = int(GW * 100), int(GH * 100)
canvas = np.zeros((CVH, CVW, 4), np.float32)
def paste(th, M):
    src = np.float32([[0, 0], [CWc, 0], [CWc, CHc]]); dst = np.float32([[M(q)[0] * 100, (GH - M(q)[1]) * 100] for q in [(0, 0), (CWc, 0), (CWc, CHc)]])
    A = cv2.getAffineTransform(src, dst)
    rgb = cv2.warpAffine(th, A, (CVW, CVH)).astype(np.float32) / 255
    m = cv2.warpAffine(np.full((CHc, CWc), 255, np.uint8), A, (CVW, CVH)).astype(np.float32) / 255
    canvas[..., :3] = rgb * m[..., None] + canvas[..., :3] * (1 - m[..., None]); canvas[..., 3] = np.maximum(canvas[..., 3], m)
paste(th1, M1); paste(th2, M2)
ax.imshow(canvas, extent=(0, GW, 0, GH), zorder=3, interpolation="bilinear", aspect="auto")
ax.set_xlim(0, GW); ax.set_ylim(0, GH)
for M in (M1, M2): ax.add_patch(Polygon(quadpts(M), closed=True, fc="none", ec="#4a6fa5", lw=2.5, zorder=4))

def L(A, B, **k): ax.plot([A[0], B[0]], [A[1], B[1]], **k)
BL = "#1f3b73"
L(C1, C2, color=BL, lw=3, zorder=2)
L(C1, Xd, color="k", ls=(0, (1.5, 3)), lw=2, zorder=2); L(C2, Xd, color="k", ls=(0, (1.5, 3)), lw=2, zorder=2)
# linhas epipolares: dentro da miniatura (forte) e prolongamento ate o epipolo (tracejado)
for (pt, dv, e), M, th in zip(epi, (M1, M2), (th1, th2)):
    L(pt - 0.0 * dv, e, color="#d00000", lw=2.2, ls=(0, (5, 3)), zorder=5)
    # trecho dentro do quadro (forte)
    ts = np.linspace(-6, 6, 4000); pts = pt[None] + ts[:, None] * dv[None]
    Minv = np.linalg.inv(np.array([[ (M((1, 0)) - M((0, 0)))[0], (M((0, 1)) - M((0, 0)))[0] ], [ (M((1, 0)) - M((0, 0)))[1], (M((0, 1)) - M((0, 0)))[1] ]]))
    uv = (Minv @ (pts - M((0, 0))).T).T
    ok = (uv[:, 0] >= 0) & (uv[:, 0] <= CWc) & (uv[:, 1] >= 0) & (uv[:, 1] <= CHc)
    q = pts[ok]; L(q[0], q[-1], color="#d00000", lw=4.5, zorder=5)
    ax.plot(*e, "o", ms=14, mfc="#ffe14a", mec="k", mew=2, zorder=7)
ax.plot(*Xd, "o", ms=17, color="k", zorder=7)
for C in (C1, C2): ax.plot(*C, "o", ms=15, mfc=BL, mec="w", mew=2, zorder=7)
for M, xp in ((M1, xp1), (M2, xp2)): ax.plot(*M(xp), "o", ms=12, mfc="#d00000", mec="k", mew=2, zorder=7)

def T(p, s, dx=0, dy=0, fs=17, **k):
    ax.text(p[0] + dx, p[1] + dy, s, fontsize=fs, zorder=9, ha=k.pop("ha", "center"), va=k.pop("va", "center"),
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2), **k)
T(Xd, "$X$ (ponto 3D)", 0, 0.65, fs=19)
T(C1, "$C_1$ (centro óptico 1)", 0, -0.7, fs=17); T(C2, "$C_2$ (centro óptico 2)", 0, -0.7, fs=17)
T(0.3 * C1 + 0.3 * C2 + 0.4 * Xd + np.array([0, 0.3]), "plano epipolar", fs=20, color="#2e7d12")
T((C1 + C2) / 2, f"linha de base  $b$ = {fm(base)} cm", 0, -0.85, fs=19, color=BL)
# rotulos x1, x2 (ao lado do ponto, acima)
T(M1(xp1), "$x_1$", -0.35, 0.45, fs=18); T(M2(xp2), "$x_2$", 0.35, 0.45, fs=18)
# nomes das imagens e linha epipolar
for M, n in ((M1, NA), (M2, NB)):
    q = quadpts(M); T(q[0] if M is M1 else q[1], f"imagem {n[-4:]}", 0.0, 0.5, fs=17, color="#34527d", ha="left" if M is M1 else "right")
for (pt, dv, e), M, sg in zip(epi, (M1, M2), (+1, -1)):
    q = quadpts(M); ex = q[1] if sg > 0 else q[0]
    pass
for (pt, dv, e), sg in zip(epi, (+1, -1)):
    T(pt + 0.72 * (e - pt) + np.array([sg * 1.3, 0.55]), "linha epipolar", fs=15, color="#d00000")
# epipolos e caixas com valores reais
T(E1d, "$e_1$", -0.55, 0.6, fs=20)
T(E2d, "$e_2$", 0.55, 0.6, fs=20)
box = dict(fs=15, va="top")
T(np.array([E1d[0] + 0.2, 2.35]), f"epipolo $e_1$ = ({fm(e1h[0], 0)}; {fm(e1h[1], 0)}) px\nfora do quadro {W}x{H} px", ha="right", **box)
T(np.array([E2d[0] - 0.2, 2.35]), f"epipolo $e_2$ = ({fm(e2h[0], 0)}; {fm(e2h[1], 0)}) px\nfora do quadro {W}x{H} px", ha="left", **box)
ax.text(GW / 2, 0.2, "Esquema fora de escala (posições de desenho livres). Dados reais: cozinha, imagens "
        f"{NA[-4:]} e {NB[-4:]}.\nMiniaturas: recorte de 560x340 px em torno de $x_1$ e $x_2$; linha epipolar: contínua dentro do recorte, tracejada até o epipolo.", fontsize=14, ha="center", va="bottom", color="#333")
fig.savefig(OUT, dpi=200); print("ok")
