"""G1b: pipeline completo (cópia estendida de G1_pipeline_sfm.py): trilha esparsa (SfM, 6 painéis do G1), trilha densa e
malha (mapas de profundidade, fusão, limpeza, Poisson, detalhe em wireframe, reconstrução final = painel 7 do G1) e, no fim,
comparação com o LiDAR. Só dados da cozinha. Layout em serpente: 4 colunas x 3 fileiras + faixa larga do LiDAR.
Rodar: /home/moliveira/Documents/3d_indoor_environment/.venv/bin/python G1b_pipeline_completo.py
  (G1B_CACHE=<pasta> guarda os renders pesados entre execuções; opcional)
Saída: ../G1b_pipeline_completo.png (15 pol, 250 dpi). Só leitura em R.
Fontes (R/cozinha/): as do G1 (artifacts/config_cozinha/*.csv, frames/all_stride10, results/config_cozinha/{sfm,scale,planes,
clean/sparse,matching,epipolar,exports/colmap/sparse/0}) e mais, em results/config_cozinha/:
  dense/patch_match/ws/stereo/depth_maps/*.geometric.bin (2.325 mapas), dense/undistorted/images, dense/patch_match/points.npy
  (98.083.833 pontos, mmap, subamostra 1 a cada 30), clean/patch_match/points.npy, malha_limpa/{nuvem_limpa.ply,malha_limpa.ply,
  resumo.json,render_malha_limpa.jpg}, report_filtragem/{poisson_q0.00.ply,poisson_q0.02.ply,filtragem.csv}, planes/floor.npy,
  mesh/room_fill/T_room.npy; LiDAR: R/cozinha_lidar/fused/mesh_closed_2M.ply e
  3d_indoor_environment_report/outputs/t5_lidar/run/lidar/T_clean_patch_match.npy; métricas: notas/t5_lidar.md.
Fatores: escala 29,3745 cm/unidade (mapas de profundidade e nuvem densa bruta)."""
import csv, os, sys, json, glob
import numpy as np, cv2, pycolmap, open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection, PolyCollection
from open3d.visualization import rendering as o3r
from PIL import Image

R = "/home/moliveira/Documents/3d_indoor_environment/"
A = R + "cozinha/artifacts/config_cozinha/"
RUN = R + "cozinha/results/config_cozinha/"
FR = R + "cozinha/frames/all_stride10/"
DM = RUN + "dense/patch_match/ws/stereo/depth_maps/"
UND = RUN + "dense/undistorted/images/"
REP = "/home/moliveira/Documents/report_3d_indoor_environment/"
sys.path.insert(0, REP + "images/report/scripts")
sys.dont_write_bytecode = True
import profundidade as PR      # read_map, recon_depth, lidar_depth (G7), só leitura
plt.rcParams.update({"font.family": ["DejaVu Sans"], "pdf.fonttype": 42})
rd = lambda p: list(csv.DictReader(open(p)))
def br(x, d=0): return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
def frame(n): return np.asarray(Image.open(FR + f"{n}.png").convert("RGB"))
def gray(n): return cv2.imread(FR + f"{n}.png", cv2.IMREAD_GRAYSCALE)   # como src/detectores.py
CACHE = os.environ.get("G1B_CACHE")


def cached(nome, fn):
    if CACHE:
        p = os.path.join(CACHE, nome + ".npz")
        if os.path.exists(p): return dict(np.load(p, allow_pickle=True))
        d = fn(); os.makedirs(CACHE, exist_ok=True); np.savez(p, **d); return d
    return fn()


PAR = "img_0774__img_0790"
n_q = int(rd(A + "extraction_metrics.csv")[0]["n_selected"])
sift = [r for r in rd(A + "detector_choice.csv") if r["detector"] == "sift"][0]
mm = [r for r in rd(A + "matching_metrics.csv") if r["detector"] == "sift" and r["pair"] == PAR][0]
sfm = rd(A + "sfm_metrics.csv")[0]
RS = json.load(open(RUN + "malha_limpa/resumo.json"))
FILT = rd(RUN + "report_filtragem/filtragem.csv")
N_DM = len(glob.glob(DM + "*.geometric.bin"))
N_RAW = np.load(RUN + "dense/patch_match/points.npy", mmap_mode="r").shape[0]
NV_POI = int([r for r in FILT if r["etapa"] == "poisson"][0]["antes"])
cap = {
 1: f"{br(n_q)} quadros",
 2: f"SIFT, score {br(float(sift['score']),3)}",
 3: f"{mm['inliers']} de {mm['n_matches_lowe']} inliers",
 4: f"{br(int(sfm['n_registered']))} poses",
 5: "quina do livro",
 6: f"erro de reprojeção médio {br(float(np.load(RUN + 'sfm/point_error.npy').mean()), 3)} px",
 7: f"{br(RS['final'][1] / 1e6, 2)} M faces, sem teto e sem exterior",
 "d1": f"{br(N_DM)} mapas de profundidade",
 "d2": f"nuvem densa bruta, {br(N_RAW / 1e6, 1)} M pontos",
 "d3": f"{br(RS['nuvem']['espurios'])} espúrios removidos ({br(100 * RS['nuvem']['espurios'] / RS['nuvem']['total'], 1)}%)",
 "d4": f"{br(NV_POI / 1e6, 2)} M vértices, antes do corte",
 "lidar": "Chamfer 3,82 cm    F-score 0,79 a 5 cm    93,5% dos pontos a até 5 cm",   # notas/t5_lidar.md
}

HERE = os.path.dirname(os.path.abspath(__file__)); IMG = os.path.dirname(HERE) + "/"
DPI = 250; ESC = 29.3745
W = 15.0
m_, g = 0.06, 0.45                       # margem e espaço entre painéis
LEG = 0.42                               # faixa da legenda das trilhas
hh = 3.15                                # altura das fileiras 1 a 3
h4 = 2.75                                # altura da faixa do LiDAR
H = LEG + 3 * hh + h4 + 3 * g + m_
fig = plt.figure(figsize=(W, H), dpi=DPI)
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis("off"); ov.set_zorder(0)
TH, CH = 0.42, 0.36      # faixa do título e da legenda curta (pol.)
FT, FC, FS = 13.5, 11.5, 10.5           # fontes: título, legenda curta, textos internos
snap = lambda v: round(v * DPI) / DPI   # eixos alinhados à grade de pixels: raster 1:1 no PNG
TINT = {"sfm": "#dbe8f5", "densa": "#fde4cf", "lidar": "#dcefd3"}


def panel(n, title, x, y, w, h, draw, trilha):
    ov.add_patch(Rectangle((x, y), w, h, fc="white", ec="none", zorder=1))
    ov.add_patch(Rectangle((x, y + h - TH), w, TH, fc=TINT[trilha], ec="none", zorder=1))
    ov.add_patch(Rectangle((x, y), w, h, fill=False, ec="black", lw=0.8, zorder=3))
    ov.text(x + w / 2, y + h - TH / 2, title, ha="center", va="center", fontsize=FT, zorder=3)
    ov.text(x + w / 2, y + CH / 2 + 0.03, cap[n], ha="center", va="center", fontsize=FC, zorder=3)
    ax_, ay, aw, ah = snap(x + 0.05), snap(y + CH + 0.05), snap(w - 0.1), snap(h - TH - CH - 0.1)
    ax = fig.add_axes([ax_ / W, ay / H, aw / W, ah / H]); ax.set_zorder(2); ax.axis("off")
    ax.pw, ax.ph = round(aw * DPI), round(ah * DPI)     # tamanho do painel em px
    ax.iw, ax.ih = aw, ah                               # e em polegadas
    ax.fx, ax.fy = ax_, ay                              # posição na figura (pol.)
    draw(ax); return ax


def resize(im, w, h):
    return np.asarray(Image.fromarray(im).resize((w, h), Image.LANCZOS))


def raster(ax, im, box=None, fit="cover", bg=255):
    """Mostra im (resolução cheia) no painel. box = (x0, y0, x1, y1) em px da fonte; 'cover' ajusta box à razão
    do painel cortando, 'contain' completa com bg. A região é reduzida com LANCZOS direto para o tamanho do painel
    e os eixos ficam em px da fonte, para desenhar sobreposições vetoriais nas coordenadas originais."""
    hh_, ww = im.shape[:2]
    x0, y0, x1, y1 = box or (0, 0, ww, hh_)
    r = ax.pw / ax.ph; cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if (bw / bh > r) == (fit == "cover"): bw = bh * r
    else: bh = bw / r
    if fit == "cover":   # mantém dentro da imagem
        cx = min(max(cx, bw / 2), ww - bw / 2); cy = min(max(cy, bh / 2), hh_ - bh / 2)
    x0, x1, y0, y1 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
    X0, Y0, X1, Y1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
    can = np.full((Y1 - Y0, X1 - X0, 3), bg, np.uint8)
    sx0, sy0, sx1, sy1 = max(X0, 0), max(Y0, 0), min(X1, ww), min(Y1, hh_)
    can[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = im[sy0:sy1, sx0:sx1]
    ax.imshow(resize(can, ax.pw, ax.ph), extent=(X0, X1, Y1, Y0), interpolation="none")
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.axis("off")


def bbox_conteudo(im, pad=12, bg=255):
    ys, xs = np.where(np.abs(im.astype(int) - bg).sum(-1) > 30)
    return (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.shape[1]), min(ys.max() + pad, im.shape[0]))


def arrow(p, q):
    ov.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=17, lw=3.0, color="black", zorder=4, shrinkA=0, shrinkB=0))


def cmap_bad(nome):
    c = plt.get_cmap(nome).copy(); c.set_bad("#222222"); return c


def colorbar_h(ax, mappable, rect, ticks, labels, rotulo, fs=FS):
    cax = ax.inset_axes(rect); cb = plt.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.set_ticks(ticks); cb.ax.set_xticklabels(labels, fontsize=fs)
    cb.ax.tick_params(length=2, pad=1); cb.outline.set_linewidth(0.5)
    cax.text(1.04, 0.5, rotulo, transform=cax.transAxes, fontsize=fs, va="center")


# ================================================================ renders 3D (Open3D, câmera comum de render_malha_limpa.py)
_ml = o3d.io.read_triangle_mesh(RUN + "malha_limpa/malha_limpa.ply")
_pl = np.load(RUN + "planes/floor.npy"); UP = _pl[:3] / np.linalg.norm(_pl[:3])
CTR = np.asarray(_ml.vertices).mean(0)
_base = np.cross(UP, [1, 0, 0]); _base /= np.linalg.norm(_base)
EYE = CTR + 640.0 * (np.cos(np.radians(55)) * _base + np.sin(np.radians(55)) * UP)   # AZ 0, EL 55, DIST 640
RW, RH = 2000, 1640                                                                   # 2x o render de conferência
alt = lambda v: v @ _pl[:3] + _pl[3]                                                  # cm acima do piso


def o3d_render(geoms):
    """geoms = [(geometria, material)]. 1o render sai preto: renderiza duas vezes. Renderiza com fundo preto e com fundo
    branco; onde os dois diferem é fundo (vira branco puro), o resto vem do render de fundo preto (malha_antes_depois.py)."""
    out = []
    for bg in ([0, 0, 0, 1], [1, 1, 1, 1]):
        r = o3r.OffscreenRenderer(RW, RH)
        for i, (gm, mat) in enumerate(geoms): r.scene.add_geometry(f"g{i}", gm, mat)
        r.scene.show_skybox(False); r.scene.set_background(bg)
        r.scene.set_lighting(r.scene.LightingProfile.NO_SHADOWS, [0.3, -0.5, -0.8])
        r.setup_camera(55, CTR, EYE, UP)
        r.render_to_image(); out.append(np.asarray(r.render_to_image())[..., :3].astype(int)); del r
    preto, branco = out
    im = preto.copy(); im[np.abs(branco - preto).sum(-1) > 300] = 255
    return im.astype(np.uint8)


def mat_pts(ps):
    m = o3r.MaterialRecord(); m.shader = "defaultUnlit"; m.point_size = ps; return m


def nuvem(p, c):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)); pc.colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1) ** 2.2)
    return pc   # cor sRGB -> linear (o shader unlit aplica a curva sRGB na saída)


CORTE_TETO = 255.0      # cm acima do piso, só na visualização (como malha_antes_depois.py)


def _bruta():
    a = np.load(RUN + "dense/patch_match/points.npy", mmap_mode="r"); c = np.load(RUN + "dense/patch_match/colors.npy", mmap_mode="r")
    idx = np.arange(0, len(a), 30)
    p = np.asarray(a[idx], np.float64) * ESC; col = np.asarray(c[idx], np.float64) / 255
    k = alt(p) <= CORTE_TETO
    return {"im": o3d_render([(nuvem(p[k], col[k]), mat_pts(2.5))]), "n": len(idx)}


def _limpeza():
    p = np.load(RUN + "clean/patch_match/points.npy").astype(np.float64)
    q = np.asarray(o3d.io.read_point_cloud(RUN + "malha_limpa/nuvem_limpa.ply").points)
    vv = lambda x: np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * 3))).ravel()
    keep = np.isin(vv(p), vv(q))
    assert (~keep).sum() == RS["nuvem"]["espurios"], (~keep).sum()
    k = alt(p) <= CORTE_TETO                       # espúrios acima do teto ficariam escondendo a sala
    col = np.load(RUN + "clean/patch_match/colors.npy").astype(np.float64); col = col / 255 if col.max() > 1.5 else col
    lum = col @ [0.299, 0.587, 0.114]                # verde modulado pelo brilho da cor real, para ler a cena
    c = np.where(keep[:, None], (0.35 + 0.65 * lum)[:, None] * [0.30, 0.72, 0.30], [0.85, 0.10, 0.10])
    g1 = nuvem(p[k & keep], c[k & keep]); g2 = nuvem(p[k & ~keep], c[k & ~keep])
    return {"im": o3d_render([(g1, mat_pts(2.0)), (g2, mat_pts(3.0))])}


def _poisson():
    m0 = o3d.io.read_triangle_mesh(RUN + "report_filtragem/poisson_q0.00.ply")
    v1 = np.asarray(o3d.io.read_triangle_mesh(RUN + "report_filtragem/poisson_q0.02.ply").vertices)
    v0, f0 = np.asarray(m0.vertices), np.asarray(m0.triangles)
    assert len(v0) == NV_POI
    vv = lambda x: np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * 3))).ravel()
    keep = np.isin(vv(v0), vv(v1))
    c = np.where(keep[:, None], np.asarray(m0.vertex_colors), [0.9, 0.05, 0.05])
    sel = alt(v0)[f0].mean(1) <= CORTE_TETO
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v0), o3d.utility.Vector3iVector(f0[sel]))
    m.vertex_colors = o3d.utility.Vector3dVector(c); m.remove_unreferenced_vertices(); m.compute_vertex_normals()
    mat = o3r.MaterialRecord(); mat.shader = "defaultLit"
    return {"im": o3d_render([(m, mat)])}


# ================================================================ 1 quadros
def mosaico(ax):
    """9 quadros 16:9 em 3 x 3, espalhados ao longo da sequência."""
    ids = [60, 720, 800, 1110, 1140, 1320, 1560, 1950, 2040]
    gp = round(0.05 * DPI)
    tw = (ax.pw - 2 * gp) // 3; th = tw * 9 // 16
    can = np.full((ax.ph, ax.pw, 3), 255, np.uint8)
    top = (ax.ph - (3 * th + 2 * gp)) // 2; lef = (ax.pw - (3 * tw + 2 * gp)) // 2
    for j, k in enumerate(ids):
        px, py = lef + (j % 3) * (tw + gp), top + (j // 3) * (th + gp)
        can[py:py + th, px:px + tw] = resize(frame(f"img_{k:04d}"), tw, th)
    ax.imshow(can, extent=(0, ax.pw, ax.ph, 0), interpolation="none")
    ax.set_xlim(0, ax.pw); ax.set_ylim(ax.ph, 0); ax.axis("off")


# ================================================================ 2 SIFT
def sift_kp(ax):
    """SIFT 5000 (src/detectores.py) redetectado no quadro original img_1110; pontos vetoriais."""
    f = frame("img_1110")
    kp = cv2.SIFT_create(nfeatures=5000).detect(gray("img_1110"), None)
    n_ok = [int(r["n_keypoints"]) for r in rd(A + "detection_metrics.csv") if r["detector"] == "sift" and r["frame"] == "img_1110"][0]
    assert len(kp) == n_ok, (len(kp), n_ok)
    raster(ax, f)
    p = np.array([k.pt for k in kp])
    ax.scatter(p[:, 0], p[:, 1], s=0.6, c="#00ff00", linewidths=0, zorder=3)


# ================================================================ 3 correspondências
def corresp(ax):
    """Mesmo par, máscaras e amostragem de matches_epilines.py (a)."""
    A_, B_ = PAR.split("__")
    rgb = [frame(A_), frame(B_)]
    kps = [cv2.SIFT_create(nfeatures=5000).detect(gray(n), None) for n in (A_, B_)]
    cross = np.load(RUN + f"matching/sift/{PAR}/matches_cross.npy"); inl = np.load(RUN + f"epipolar/sift/{PAR}/inlier_mask.npy")
    assert len(cross) == int(mm["n_matches_lowe"]) and inl.sum() == int(mm["inliers"])
    pa = np.float32([kps[0][int(i)].pt for i in cross[:, 0]]); pb = np.float32([kps[1][int(i)].pt for i in cross[:, 1]])
    y0, y1 = 330, 990; Hc = y1 - y0; GAP = 40
    st = np.vstack([rgb[0][y0:y1], np.full((GAP, 1920, 3), 255, np.uint8), rgb[1][y0:y1]])
    in_crop = lambda p: (p[:, 1] >= y0) & (p[:, 1] < y1)
    vis = in_crop(pa) & in_crop(pb)
    idx_in = np.flatnonzero(inl & vis); idx_out = np.flatnonzero(~inl & vis)
    rng = np.random.default_rng(0)
    idx_in_s = idx_in[np.argsort(pa[idx_in, 0])][np.linspace(0, len(idx_in) - 1, min(60, len(idx_in))).astype(int)]
    idx_out_s = rng.choice(idx_out, min(30, len(idx_out)), replace=False)
    raster(ax, st, fit="contain")
    ta = lambda p: np.c_[p[:, 0], p[:, 1] - y0]; tb = lambda p: np.c_[p[:, 0], p[:, 1] - y0 + Hc + GAP]
    for idx, c, lw in ((idx_out_s, "#ff2a2a", 0.4), (idx_in_s, "#22ee44", 0.5)):
        ax.add_collection(LineCollection(np.stack([ta(pa[idx]), tb(pb[idx])], 1), colors=c, linewidths=lw, alpha=0.9))
        q = np.r_[ta(pa[idx]), tb(pb[idx])]
        ax.scatter(q[:, 0], q[:, 1], s=2, c=c, linewidths=0, zorder=3)


# ================================================================ geometria comum (vista de cima, como planta.py)
T = np.load(RUN + "planes/T_floor.npy"); fl = lambda p: p @ T[:3, :3].T + T[:3, 3]
Hm = lambda xy: np.c_[xy[:, 1], -xy[:, 0]]
_lab = np.load(RUN + "planes/labels.npy"); _Ps = fl(np.load(RUN + "clean/sparse/points.npy"))
_w = _Ps[_lab >= 2, :2].astype(np.float32)
_a = np.deg2rad(cv2.minAreaRect(_w)[2]); _rot = np.array([[np.cos(_a), np.sin(_a)], [-np.sin(_a), np.cos(_a)]])
_loc = _w @ _rot.T; _lo, _hi = np.percentile(_loc, [1, 99], axis=0)
for k in range(2):
    for e in (_lo, _hi): e[k] = np.median(_loc[np.abs(_loc[:, k] - e[k]) <= 5.0, k])
(X0_, Y0_), (X1_, Y1_) = _lo, _hi


def fit_view(ax, M=20):
    xl, xr, yb, yt_ = Y0_ - M, Y1_ + M, -X1_ - M, -X0_ + M
    r = ax.pw / ax.ph; cx, cy = (xl + xr) / 2, (yb + yt_) / 2; bw, bh = xr - xl, yt_ - yb
    if bw / bh > r: bh = bw / r
    else: bw = bh * r
    ax.set_xlim(cx - bw / 2, cx + bw / 2); ax.set_ylim(cy - bh / 2, cy + bh / 2); ax.axis("off")


def contorno(ax):
    ax.add_patch(Rectangle((Y0_, -X1_), Y1_ - Y0_, X1_ - X0_, fill=False, ec="0.35", lw=0.8, ls=":"))


# ================================================================ 4 poses (trajetória + frustums)
def poses(ax):
    C = fl(np.load(RUN + "scale/centers_cm.npy")); n = len(C); assert n == int(sfm["n_registered"])
    Rw = np.load(RUN + "sfm/cam_R.npy")        # mundo -> câmera
    P = _Ps; col = np.load(RUN + "clean/sparse/colors.npy"); col = col / 255 if col.max() > 1 else col
    keep = (P[:, 2] > 5) & (P[:, 2] < 270.0); Q = Hm(P[keep, :2]); Cs = Hm(C[:, :2])
    ax.scatter(Q[:, 0], Q[:, 1], s=0.3, c=np.clip(col[keep], 0, 1), alpha=0.55, marker=".", linewidths=0, rasterized=True)
    contorno(ax)
    ax.plot(Cs[:, 0], Cs[:, 1], color="0.2", lw=0.5, alpha=0.6, zorder=3)
    ax.scatter(Cs[:, 0], Cs[:, 1], c=np.arange(n), cmap="plasma", s=1.4, linewidths=0, zorder=4)
    for i in range(0, n, 110):                                              # frustums em planta
        d = T[:3, :3] @ Rw[i][2]; d = Hm(d[None, :2])[0]; d /= (np.linalg.norm(d) + 1e-9)
        nrm = np.array([-d[1], d[0]]); c = Cs[i]; L, wd = 26, 16
        pts = np.array([c, c + L * d + wd * nrm, c + L * d - wd * nrm, c])
        ax.plot(pts[:, 0], pts[:, 1], color="black", lw=0.9, zorder=5)
        ax.fill(pts[:, 0], pts[:, 1], color="#ffd400", alpha=0.35, zorder=5)
    ax.scatter(*Cs[0], s=45, marker="o", fc="#00c000", ec="k", lw=0.6, zorder=6)
    ax.scatter(*Cs[-1], s=45, marker="s", fc="#e00000", ec="k", lw=0.6, zorder=6)
    fit_view(ax)


# ================================================================ 5 triangulação
def triang(ax):
    """Ponto 3046041 (quina do livro) e seus keypoints observados em img_1150/img_1154 (triangulacao.py):
    recortes 800x600 dos quadros originais em torno do keypoint, raios vetoriais até o ponto 3D."""
    PID, IDS, NOMES, CW, CHp = 3046041, (1151, 1155), ("img_1150", "img_1154"), 800, 600
    rec = pycolmap.Reconstruction(RUN + "exports/colmap/sparse/0"); p = rec.points3D[PID]
    aw, ah = ax.iw, ax.ih
    ax.set_xlim(0, aw); ax.set_ylim(0, ah); ax.axis("off")
    fw = 0.36 * aw; fh = fw * CHp / CW; fy = 0.25
    X = np.array([aw / 2, ah - 0.42])
    cor = {IDS[0]: "#1f77b4", IDS[1]: "#d95f02"}
    for i, nome, fx in zip(IDS, NOMES, (0.15, aw - fw - 0.15)):
        im = rec.images[i]; assert im.name == nome + ".png"
        e = [q for q in p.track.elements if q.image_id == i][0]
        ox, oy = im.points2D[e.point2D_idx].xy
        cx0 = int(np.clip(round(ox - CW / 2), 0, 1920 - CW)); cy0 = int(np.clip(round(oy - CHp / 2), 0, 1080 - CHp))
        crop = resize(frame(nome)[cy0:cy0 + CHp, cx0:cx0 + CW], round(fw * DPI), round(fh * DPI))
        ax.imshow(crop, extent=(fx, fx + fw, fy, fy + fh), interpolation="none", zorder=1)
        ax.add_patch(Rectangle((fx, fy), fw, fh, fill=False, ec="black", lw=0.6, zorder=3))
        kp = np.array([fx + (ox - cx0) / CW * fw, fy + (1 - (oy - cy0) / CHp) * fh])
        ax.plot([kp[0], X[0]], [kp[1], X[1]], color=cor[i], lw=1.2, zorder=2, solid_capstyle="round")
        ax.plot(*kp, "o", mfc="none", mec="#ffe600", mew=1.1, ms=6, zorder=4)
        ax.text(fx + fw / 2, fy - 0.03, "quadro 1" if i == IDS[0] else "quadro 2", ha="center", va="top", fontsize=FC)
    ax.plot(*X, "o", color="#d62728", ms=6, mec="black", mew=0.4, zorder=5)
    ax.text(X[0], X[1] + 0.07, "ponto 3D (quina do livro)", ha="center", va="bottom", fontsize=FC)


# ================================================================ 6 ajuste de feixes: erro de reprojeção por ponto
def ba(ax):
    S = RUN + "sfm/"
    P = fl(np.load(S + "points3d.npy") * ESC); err = np.load(S + "point_error.npy")
    assert len(P) == len(err) == int(sfm["n_sparse"]) and abs(err.mean() - 0.565) < 0.001, err.mean()
    keep = (P[:, 2] > 5) & (P[:, 2] < 270.0)
    Q = Hm(P[keep, :2]); e = err[keep]; o = np.argsort(e)                    # erros altos por cima
    vmax = float(np.percentile(err, 99))
    sc = ax.scatter(Q[o, 0], Q[o, 1], c=e[o], cmap="turbo", vmin=0, vmax=vmax, s=0.6, marker=".", linewidths=0, rasterized=True)
    contorno(ax); fit_view(ax, M=34)
    colorbar_h(ax, sc, [0.05, 0.07, 0.42, 0.035], [0, vmax / 2, vmax], ["0", br(vmax / 2, 1), br(vmax, 1)], "erro (px)")


# ================================================================ D1 mapas de profundidade (PatchMatch, geométrico)
VMIN, VMAX = 50, 250       # cm, como profundidade.py (G7)
DEP_FR = "img_1110"


def mapa_prof(ax):
    rgb = np.asarray(Image.open(UND + DEP_FR + ".png").convert("RGB"))
    d = PR.read_map(DM + DEP_FR + ".png.geometric.bin") * ESC; d[d <= 0] = np.nan
    hh_, ww = d.shape; assert rgb.shape[:2] == d.shape
    cmap = cmap_bad("turbo")
    dep = (cmap(plt.Normalize(VMIN, VMAX)(np.ma.masked_invalid(d)))[..., :3] * 255).astype(np.uint8)
    xs = ww // 2
    comp = np.concatenate([rgb[:, :xs], dep[:, xs:]], 1)
    # área útil: imagem em cima, barra de cores embaixo
    ax.set_xlim(0, ax.iw); ax.set_ylim(0, ax.ih); ax.axis("off")
    iw = ax.iw; ih = iw * hh_ / ww; y0 = ax.ih - ih
    ax.imshow(resize(comp, ax.pw, round(ih * DPI)), extent=(0, iw, y0, ax.ih), interpolation="none")
    ax.plot([iw / 2, iw / 2], [y0, ax.ih], color="white", lw=2.0)
    for x, t, ha in ((0.06, "quadro", "left"), (iw - 0.06, "profundidade", "right")):
        ax.text(x, ax.ih - 0.06, t, ha=ha, va="top", fontsize=FS, color="black",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(VMIN, VMAX), cmap="turbo")
    colorbar_h(ax, sm, [0.04, (y0 - 0.2) / ax.ih, 0.42, 0.05], [50, 150, 250], ["50", "150", "250"], "profundidade (cm)")


# ================================================================ D2 a D4: renders 3D
def render_panel(chave, fn, legenda=None, y0=0.87):
    def draw(ax):
        im = cached(chave, fn)["im"]
        raster(ax, im, box=bbox_conteudo(im), fit="contain")
        if legenda:
            for j, (c, t) in enumerate(legenda):
                y = y0 - 0.12 * j
                if c: ax.add_patch(Rectangle((0.02, y), 0.035, 0.08, transform=ax.transAxes, fc=c, ec="none", zorder=5))
                ax.text(0.07, y + 0.04, t, transform=ax.transAxes, fontsize=FS, va="center", zorder=5)
    return draw


# ================================================================ D5 detalhe da malha (wireframe colorido)
def _detalhe():
    """Caixa de 20 cm em torno dos potes azuis da bancada (img_1620), malha limpa sem simplificação, no referencial da sala."""
    Ti = np.linalg.inv(np.load(RUN + "mesh/room_fill/T_room.npy"))
    v = np.asarray(_ml.vertices); f = np.asarray(_ml.triangles); c = np.asarray(_ml.vertex_colors)
    q = v @ Ti[:3, :3].T + Ti[:3, 3]                 # (x, y, h) da sala
    c0 = np.array([-132.9, -130.1, 108.9]); L = DET_L / 2
    inb = np.all(np.abs(q - c0) <= L, 1)
    fs = f[inb[f].all(1)]
    rc = np.r_[np.median(q[:, :2], 0), 100.0]        # olha do meio da sala para a bancada, 34 graus de cima
    d = c0 - rc; d[2] = 0; d /= np.linalg.norm(d); d = d * np.cos(0.6) - np.array([0, 0, np.sin(0.6)])
    d /= np.linalg.norm(d); rr = np.cross(d, [0, 0, 1.0]); rr /= np.linalg.norm(rr); uu = np.cross(rr, d)
    P = q[fs] - c0
    xy = np.stack([P @ rr, P @ uu], -1); z = (P @ d).mean(1)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    fc = np.clip(c[fs].mean(1) * (0.45 + 0.55 * np.abs(n @ -d))[:, None], 0, 1)    # cor do vértice x sombreamento
    o = np.argsort(-z)                               # do fundo para a frente (painter)
    return {"xy": xy[o], "fc": fc[o], "nf": len(fs)}


DET_L = 20.0


def detalhe(ax):
    d = cached("detalhe", _detalhe)
    xy, fc = d["xy"], d["fc"]
    ax.add_collection(PolyCollection(xy, facecolors=fc, edgecolors="#151515", linewidths=0.15))
    r = ax.pw / ax.ph; bh = DET_L * 0.8; bw = bh * r
    ax.set_xlim(-bw / 2, bw / 2); ax.set_ylim(-bh / 2, bh / 2); ax.set_aspect("equal"); ax.axis("off")
    x0, y0 = -bw / 2 + 0.05 * bw, -bh / 2 + 0.07 * bh                      # barra de escala de 2 cm
    ax.add_patch(Rectangle((x0 - 0.45, y0 - 0.55), 2.9, 2.0, fc="white", ec="none", alpha=0.9, zorder=5))
    ax.plot([x0, x0 + 2], [y0, y0], color="black", lw=2.4, zorder=6, solid_capstyle="butt")
    ax.text(x0 + 1, y0 + 0.2, "2 cm", ha="center", va="bottom", fontsize=FS, zorder=6)
    cap["d5"] = f"potes na bancada, {br(int(d['nf']))} triângulos"


# ================================================================ 7 reconstrução final
def _final():
    m = o3d.geometry.TriangleMesh(_ml); m.compute_vertex_normals()
    mat = o3r.MaterialRecord(); mat.shader = "defaultLit"
    return {"im": o3d_render([(m, mat)])}


# ================================================================ LiDAR
LID_FR = "img_1740"           # quadro de profundidade_rgb_recon_lidar.jpg (G7)


def _lidar():
    b, c, ok, dd = PR.stats(LID_FR)
    T5 = np.load(PR.T_NPY); s = np.cbrt(abs(np.linalg.det(T5[:3, :3])))
    p = np.load(RUN + "clean/patch_match/points.npy").astype(np.float64)
    pl = p @ T5[:3, :3].T + T5[:3, 3]                                          # cm do LiDAR
    sc = PR.setup()["sc"]
    dist = sc.compute_distance(o3d.core.Tensor(pl.astype(np.float32))).numpy() / s   # cm do modelo
    return {"b": b, "c": c, "pl": pl.astype(np.float32), "dist": dist.astype(np.float32)}


def lidar(ax):
    d = cached("lidar", _lidar)
    b, c = d["b"], d["c"]; diff = np.where(np.isfinite(b) & np.isfinite(c), np.abs(b - c), np.nan)
    rgb = np.asarray(Image.open(UND + LID_FR + ".png").convert("RGB"))
    ax.set_xlim(0, ax.iw); ax.set_ylim(0, ax.ih); ax.axis("off")
    LBL = 0.3
    # (a) planta: nuvem densa alinhada colorida pela distância à malha LiDAR, sem teto (y do LiDAR para cima)
    pl, dist = d["pl"], d["dist"]
    k = pl[:, 1] < 120.0                                    # piso em -96 cm: corta acima de ~2,2 m
    xz = pl[k][:, [0, 2]]; e = dist[k]; o = np.argsort(e)  # distâncias altas por cima
    lo, hi = np.percentile(xz, 1, 0) - 10, np.percentile(xz, 99, 0) + 10
    if (hi - lo)[1] > (hi - lo)[0]:                         # sala deitada: lado maior na horizontal
        xz = xz[:, ::-1]; lo, hi = lo[::-1], hi[::-1]
    rp = (hi - lo)[0] / (hi - lo)[1]; GA, GI, CB = 0.16, 0.1, 1.6       # espaços e largura reservada às barras de cores
    ih = min(ax.ih - LBL - 0.02, (ax.iw - GA - 4 * GI - CB) / (rp + 4 * 1600 / 896)); iw = ih * 1600 / 896; pw_ = ih * rp
    x = (ax.iw - (pw_ + GA + 4 * iw + 4 * GI + CB)) / 2
    axa = fig.add_axes([(ax.fx + x) / W, ax.fy / H, pw_ / W, ih / H]); axa.set_zorder(3)
    axa.scatter(xz[o, 0], xz[o, 1], c=e[o], cmap="magma", vmin=0, vmax=20, s=0.7, marker=".", linewidths=0, rasterized=True)
    axa.set_xlim(lo[0], hi[0]); axa.set_ylim(lo[1], hi[1]); axa.set_xticks([]); axa.set_yticks([])
    axa.axis("off")
    ax.text(x + pw_ / 2, ih + 0.06, "nuvem densa × LiDAR (planta)", ha="center", va="bottom", fontsize=FC)
    x += pw_ + GA
    ims = [(rgb, "quadro RGB", None), (b, "profundidade reconstruída", ("turbo", VMIN, VMAX)),
           (c, "profundidade LiDAR", ("turbo", VMIN, VMAX)), (diff, "|reconstrução − LiDAR|", ("magma", 0, 20))]
    for im, t, cm in ims:
        if cm:
            cmp = cmap_bad(cm[0]); im = (cmp(plt.Normalize(cm[1], cm[2])(np.ma.masked_invalid(im)))[..., :3] * 255).astype(np.uint8)
        ax.imshow(resize(im, round(iw * DPI), round(ih * DPI)), extent=(x, x + iw, 0, ih), interpolation="none")
        ax.text(x + iw / 2, ih + 0.06, t, ha="center", va="bottom", fontsize=FC)
        x += iw + GI
    for j, (cm, v0, v1, tk, rot) in enumerate((("turbo", VMIN, VMAX, [50, 150, 250], "profundidade (cm)"),
                                                ("magma", 0, 20, [0, 10, 20], "distância (cm)"))):
        cx_ = x + 0.04 + j * 0.8
        cax = fig.add_axes([(ax.fx + cx_) / W, (ax.fy + 0.04) / H, 0.1 / W, (ih - 0.08) / H])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(v0, v1), cmap=cm), cax=cax)
        cb.set_ticks(tk); cb.ax.tick_params(labelsize=FS, length=2, pad=1); cb.outline.set_linewidth(0.5)
        cb.set_label(rot, fontsize=FS, labelpad=3)


# ================================================================ layout em serpente
xs = [m_ + i * ((W - 2 * m_ - 3 * g) / 4 + g) for i in range(4)]; wt = (W - 2 * m_ - 3 * g) / 4
y1 = H - LEG - hh; y2 = y1 - g - hh; y3 = y2 - g - hh; y4 = m_
# legenda das trilhas
lx = [("sfm", "trilha esparsa (SfM)"), ("densa", "trilha densa e malha"), ("lidar", "avaliação com o LiDAR")]
tot = 0; xl = W / 2 - 4.4
for k, t in lx:
    ov.add_patch(Rectangle((xl, H - LEG / 2 - 0.1), 0.32, 0.2, fc=TINT[k], ec="black", lw=0.6))
    ov.text(xl + 0.42, H - LEG / 2, t, fontsize=FT - 1, va="center"); xl += 0.42 + 0.115 * len(t) + 0.6

panel(1, "Quadros de entrada", xs[0], y1, wt, hh, mosaico, "sfm")
panel(2, "Extração SIFT", xs[1], y1, wt, hh, sift_kp, "sfm")
panel(3, "Correspondências", xs[2], y1, wt, hh, corresp, "sfm")
panel(4, "Estimativa das poses", xs[3], y1, wt, hh, poses, "sfm")
panel(5, "Triangulação", xs[3], y2, wt, hh, triang, "sfm")
panel(6, "Ajuste de feixes", xs[2], y2, wt, hh, ba, "sfm")
panel("d1", "Mapas de profundidade", xs[1], y2, wt, hh, mapa_prof, "densa")
panel("d2", "Fusão das profundidades", xs[0], y2, wt, hh, render_panel("bruta", _bruta), "densa")
panel("d3", "Limpeza da nuvem", xs[0], y3, wt, hh,
      render_panel("limpeza", _limpeza, [("#d91f1f", "espúrios"), ("#4cb84c", "mantidos")]), "densa")
panel("d4", "Malha de Poisson", xs[1], y3, wt, hh,
      render_panel("poisson", _poisson, [("#e60d0d", "baixa densidade")]), "densa")
cap["d5"] = ""
ax5 = panel("d5", "Detalhe da malha", xs[2], y3, wt, hh, detalhe, "densa")
ov.texts[-1].set_text(cap["d5"])
panel(7, "Reconstrução final", xs[3], y3, wt, hh, render_panel("final", _final), "densa")
panel("lidar", "Comparação com o LiDAR", m_, y4, W - 2 * m_, h4, lidar, "lidar")

# setas
ya = y1 + hh / 2
for i in range(3): arrow((xs[i] + wt + 0.03, ya), (xs[i + 1] - 0.03, ya))
arrow((xs[3] + wt / 2, y1 - 0.03), (xs[3] + wt / 2, y2 + hh + 0.03))
yb = y2 + hh / 2
for i in (3, 2, 1): arrow((xs[i] - 0.03, yb), (xs[i - 1] + wt + 0.03, yb))
arrow((xs[0] + wt / 2, y2 - 0.03), (xs[0] + wt / 2, y3 + hh + 0.03))
yc = y3 + hh / 2
for i in range(3): arrow((xs[i] + wt + 0.03, yc), (xs[i + 1] - 0.03, yc))
arrow((xs[3] + wt / 2, y3 - 0.03), (xs[3] + wt / 2, y4 + h4 + 0.03))

fig.savefig(IMG + "G1b_pipeline_completo.png", dpi=DPI, pil_kwargs={"optimize": True})
print(Image.open(IMG + "G1b_pipeline_completo.png").size, {k: v for k, v in cap.items()})
