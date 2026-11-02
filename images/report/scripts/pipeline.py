"""Figura fig:pipeline (estilo serpente): cada etapa mostra a imagem real que produz.
Rodar (só leitura em R = /home/moliveira/Documents/3d_indoor_environment):
  R/.venv/bin/python pipeline_malha.py   # uma vez: renders 3D em alta (_densa_hi.png, _malha_hi.png)
  R/.venv/bin/python pipeline.py
Saídas em images/report/: pipeline.pdf (vetorial: texto, molduras, setas, pontos SIFT, linhas de correspondência,
trajetória e raios; fotos e renders embutidos a DPI px/pol no tamanho exato de cada painel), pipeline.png e
pipeline.jpg (mesmo conteúdo, DPI px/pol).
Cada painel vem da fonte em resolução cheia: quadros PNG 1920x1080 de R/cozinha/frames/all_stride10, reduzidos com
LANCZOS direto para o tamanho final do painel.
Números (R/cozinha/artifacts/config_cozinha/): extraction_metrics.csv (2327), detector_choice.csv (SIFT 0,936),
matching_metrics.csv (666 de 765, img_0774__img_0790, sift), sfm_metrics.csv (2326 registradas, 727716 pontos),
cleanup.csv (1428041 pontos), mesh.csv (3074852 faces)."""
import csv, os, subprocess
import numpy as np, cv2, pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.collections import LineCollection
from PIL import Image

R = "/home/moliveira/Documents/3d_indoor_environment/"
A = R + "cozinha/artifacts/config_cozinha/"
RUN = R + "cozinha/results/config_cozinha/"
FR = R + "cozinha/frames/all_stride10/"
HERE = os.path.dirname(os.path.abspath(__file__)); IMG = os.path.dirname(HERE) + "/"
DPI = 450
plt.rcParams.update({"font.family": ["DejaVu Sans"], "pdf.fonttype": 42})
rd = lambda p: list(csv.DictReader(open(p)))
def br(x, d=0): return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
def frame(n): return np.asarray(Image.open(FR + f"{n}.png").convert("RGB"))
def gray(n): return cv2.imread(FR + f"{n}.png", cv2.IMREAD_GRAYSCALE)   # como src/detectores.py

PAR = "img_0774__img_0790"
n_q = int(rd(A + "extraction_metrics.csv")[0]["n_selected"])
sift = [r for r in rd(A + "detector_choice.csv") if r["detector"] == "sift"][0]
mm = [r for r in rd(A + "matching_metrics.csv") if r["detector"] == "sift" and r["pair"] == PAR][0]
sfm = rd(A + "sfm_metrics.csv")[0]
cl = [r for r in rd(A + "cleanup.csv") if r["source"] == "patch_match"][0]
mesh = rd(A + "mesh.csv")[0]
cap = {
 1: f"{br(n_q)} quadros",
 2: f"SIFT, escore {br(float(sift['score']),3)}",
 3: f"{mm['inliers']} de {mm['n_matches_lowe']} inliers",
 4: f"{br(int(sfm['n_registered']))} poses",
 5: f"{br(int(sfm['n_sparse']))} pontos 3D",
 6: f"{br(int(cl['n_after_voxel'])/1e6,2)} M pontos",
 7: f"{br(int(mesh['n_faces'])/1e6,2)} M faces",
}

W, H = 6.67, 3.0
fig = plt.figure(figsize=(W, H), dpi=DPI)
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis("off"); ov.set_zorder(0)
TH, CH = 0.2, 0.17      # faixa do título e da legenda curta (pol.)
snap = lambda v: round(v * DPI) / DPI   # eixos alinhados à grade de pixels: raster 1:1 no PNG


def panel(n, title, x, y, w, h, draw):
    ov.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=0.7, zorder=1))
    ov.text(x + w / 2, y + h - TH / 2 - 0.02, title, ha="center", va="center", fontsize=7.5, zorder=3)
    ov.text(x + w / 2, y + CH / 2 + 0.03, cap[n], ha="center", va="center", fontsize=7, zorder=3)
    ax_, ay, aw, ah = snap(x + 0.05), snap(y + CH + 0.05), snap(w - 0.1), snap(h - TH - CH - 0.1)
    ax = fig.add_axes([ax_ / W, ay / H, aw / W, ah / H]); ax.set_zorder(2); ax.axis("off")
    ax.pw, ax.ph = round(aw * DPI), round(ah * DPI)     # tamanho do painel em px
    ax.iw, ax.ih = aw, ah                               # e em polegadas
    draw(ax); return ax


def resize(im, w, h):
    return np.asarray(Image.fromarray(im).resize((w, h), Image.LANCZOS))


def raster(ax, im, box=None, fit="cover", bg=255):
    """Mostra im (resolução cheia) no painel. box = (x0, y0, x1, y1) em px da fonte; 'cover' ajusta box à razão
    do painel cortando, 'contain' completa com bg. A região é reduzida com LANCZOS direto para o tamanho do painel
    e os eixos ficam em px da fonte, para desenhar sobreposições vetoriais nas coordenadas originais."""
    hh, ww = im.shape[:2]
    x0, y0, x1, y1 = box or (0, 0, ww, hh)
    r = ax.pw / ax.ph; cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    if (bw / bh > r) == (fit == "cover"): bw = bh * r
    else: bh = bw / r
    if fit == "cover":   # mantém dentro da imagem
        cx = min(max(cx, bw / 2), ww - bw / 2); cy = min(max(cy, bh / 2), hh - bh / 2)
    x0, x1, y0, y1 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
    X0, Y0, X1, Y1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
    can = np.full((Y1 - Y0, X1 - X0, 3), bg, np.uint8)
    sx0, sy0, sx1, sy1 = max(X0, 0), max(Y0, 0), min(X1, ww), min(Y1, hh)
    can[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = im[sy0:sy1, sx0:sx1]
    ax.imshow(resize(can, ax.pw, ax.ph), extent=(X0, X1, Y1, Y0), interpolation="none")
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.axis("off")


def arrow(p, q):
    ov.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=2.0, color="black", zorder=4, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- 1 quadros
def mosaico(ax):
    """5 quadros (recorte central 4:3, sem distorcer) em 3 + 2, montados direto no tamanho final em px."""
    ids = [800, 1110, 1400, 1950, 2299]
    g = round(0.03 * DPI)
    tw = (ax.pw - 2 * g) // 3; th = tw * 3 // 4
    can = np.full((ax.ph, ax.pw, 3), 255, np.uint8)
    top = (ax.ph - (2 * th + g)) // 2
    pos = [(0, top), (tw + g, top), (2 * (tw + g), top), ((tw + g) // 2, top + th + g), ((tw + g) // 2 + tw + g, top + th + g)]
    for k, (px, py) in zip(ids, pos):
        f = frame(f"img_{k:04d}"); c0 = (1920 - 1440) // 2
        can[py:py + th, px:px + tw] = resize(f[:, c0:c0 + 1440], tw, th)
    ax.imshow(can, extent=(0, ax.pw, ax.ph, 0), interpolation="none")
    ax.set_xlim(0, ax.pw); ax.set_ylim(ax.ph, 0); ax.axis("off")


# ---------------------------------------------------------------- 2 SIFT
def sift_kp(ax):
    """SIFT 5000 (src/detectores.py) redetectado no quadro original img_1110; pontos vetoriais."""
    f = frame("img_1110")
    kp = cv2.SIFT_create(nfeatures=5000).detect(gray("img_1110"), None)
    n_ok = [int(r["n_keypoints"]) for r in rd(A + "detection_metrics.csv") if r["detector"] == "sift" and r["frame"] == "img_1110"][0]
    assert len(kp) == n_ok, (len(kp), n_ok)
    raster(ax, f)
    p = np.array([k.pt for k in kp])
    ax.scatter(p[:, 0], p[:, 1], s=0.4, c="#00ff00", linewidths=0, zorder=3)


# ---------------------------------------------------------------- 3 correspondências
def corresp(ax):
    """Mesmo par, máscaras e amostragem de matches_epilines.py (a): matches_cross.npy e inlier_mask.npy salvos,
    keypoints SIFT 5000 redetectados (determinísticos), recorte y 330:990, 60 inliers e 30 outliers (rng 0)."""
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
    for idx, c, lw in ((idx_out_s, "#ff2a2a", 0.35), (idx_in_s, "#22ee44", 0.45)):
        ax.add_collection(LineCollection(np.stack([ta(pa[idx]), tb(pb[idx])], 1), colors=c, linewidths=lw, alpha=0.9))
        q = np.r_[ta(pa[idx]), tb(pb[idx])]
        ax.scatter(q[:, 0], q[:, 1], s=1.5, c=c, linewidths=0, zorder=3)


# ---------------------------------------------------------------- 4 poses
def poses(ax):
    """Trajetória vista de cima (mesmos eixos/recorte de planta.py): esparsa sem teto + 2.326 centros."""
    T = np.load(RUN + "planes/T_floor.npy"); fl = lambda p: p @ T[:3, :3].T + T[:3, 3]
    P = fl(np.load(RUN + "clean/sparse/points.npy")); col = np.load(RUN + "clean/sparse/colors.npy")
    if col.max() > 1: col = col / 255
    lab = np.load(RUN + "planes/labels.npy"); C = fl(np.load(RUN + "scale/centers_cm.npy")); n = len(C)
    assert n == int(sfm["n_registered"])
    w = P[lab >= 2, :2].astype(np.float32)
    ang = cv2.minAreaRect(w)[2]; a = np.deg2rad(ang); rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    loc = w @ rot.T; lo, hi = np.percentile(loc, [1, 99], axis=0)
    for k in range(2):
        for e in (lo, hi): e[k] = np.median(loc[np.abs(loc[:, k] - e[k]) <= 5.0, k])
    (x0, y0), (x1, y1) = lo, hi
    Hm = lambda xy: np.c_[xy[:, 1], -xy[:, 0]]
    keep = (P[:, 2] > 5) & (P[:, 2] < 270.0)
    Q = Hm(P[keep, :2]); Cs = Hm(C[:, :2])
    ax.scatter(Q[:, 0], Q[:, 1], s=0.12, c=np.clip(col[keep], 0, 1), marker=".", linewidths=0, rasterized=True)
    ax.add_patch(Rectangle((y0, -x1), y1 - y0, x1 - x0, fill=False, ec="0.35", lw=0.5, ls=":"))
    ax.plot(Cs[:, 0], Cs[:, 1], color="0.2", lw=0.25, alpha=0.5, zorder=3)
    ax.scatter(Cs[:, 0], Cs[:, 1], c=np.arange(n), cmap="plasma", s=0.35, linewidths=0, zorder=4)
    ax.scatter(*Cs[0], s=9, marker="o", fc="#00c000", ec="k", lw=0.4, zorder=6)
    ax.scatter(*Cs[-1], s=9, marker="s", fc="#e00000", ec="k", lw=0.4, zorder=6)
    M = 20; xl, xr, yb, yt = y0 - M, y1 + M, -x1 - M, -x0 + M
    r = ax.pw / ax.ph; cx, cy = (xl + xr) / 2, (yb + yt) / 2; bw, bh = xr - xl, yt - yb
    if bw / bh > r: bh = bw / r
    else: bw = bh * r
    ax.set_xlim(cx - bw / 2, cx + bw / 2); ax.set_ylim(cy - bh / 2, cy + bh / 2); ax.axis("off")


# ---------------------------------------------------------------- 5 triangulação
def triang(ax):
    """Ponto 3046041 (quina do livro) e seus keypoints observados em img_1150/img_1154 (triangulacao.py):
    recortes 800x600 dos quadros originais em torno do keypoint, raios vetoriais até o ponto 3D."""
    PID, IDS, NOMES, CW, CHp = 3046041, (1151, 1155), ("img_1150", "img_1154"), 800, 600
    rec = pycolmap.Reconstruction(RUN + "exports/colmap/sparse/0"); p = rec.points3D[PID]
    aw, ah = ax.iw, ax.ih
    ax.set_xlim(0, aw); ax.set_ylim(0, ah); ax.axis("off")
    fw = 0.66; fh = fw * CHp / CW; fy = 0.16
    X = np.array([aw / 2, 0.80])
    cor = {IDS[0]: "#1f77b4", IDS[1]: "#d95f02"}
    for i, nome, fx in zip(IDS, NOMES, (0.02, aw - 0.02 - fw)):
        im = rec.images[i]; assert im.name == nome + ".png"
        e = [q for q in p.track.elements if q.image_id == i][0]
        ox, oy = im.points2D[e.point2D_idx].xy
        cx0 = int(np.clip(round(ox - CW / 2), 0, 1920 - CW)); cy0 = int(np.clip(round(oy - CHp / 2), 0, 1080 - CHp))
        crop = resize(frame(nome)[cy0:cy0 + CHp, cx0:cx0 + CW], round(fw * DPI), round(fh * DPI))
        ax.imshow(crop, extent=(fx, fx + fw, fy, fy + fh), interpolation="none", zorder=1)
        ax.add_patch(Rectangle((fx, fy), fw, fh, fill=False, ec="black", lw=0.6, zorder=3))
        kp = np.array([fx + (ox - cx0) / CW * fw, fy + (1 - (oy - cy0) / CHp) * fh])
        ax.plot([kp[0], X[0]], [kp[1], X[1]], color=cor[i], lw=1.1, zorder=2, solid_capstyle="round")
        ax.plot(*kp, "o", mfc="none", mec="#ffe600", mew=1.0, ms=5, zorder=4)
        ax.text(fx + fw / 2, fy - 0.03, "quadro 1" if i == IDS[0] else "quadro 2", ha="center", va="top", fontsize=7)
    ax.plot(*X, "o", color="#d62728", ms=5, mec="black", mew=0.4, zorder=5)
    ax.text(X[0], X[1] + 0.06, "ponto 3D (quina do livro)", ha="center", va="bottom", fontsize=7)


# ---------------------------------------------------------------- 6 e 7 (renders de pipeline_malha.py)
def densa(ax):
    im = np.asarray(Image.open(HERE + "/_densa_hi.png").convert("RGB"))
    s = im.shape[1] / 1400   # mesmo recorte do painel anterior: (120, 30, 1290, 1030) na miniatura de 1400 px
    raster(ax, im, (120 * s, 30 * s, 1290 * s, 1030 * s), fit="contain")


def malha(ax):
    raster(ax, np.asarray(Image.open(HERE + "/_malha_hi.png").convert("RGB")), fit="contain")


# fileira de cima (esquerda para direita)
yt, hh = 1.72, 1.25
y2, h2 = 0.03, 1.5
x = 0.05; ws = [1.45, 1.2, 1.75, 1.28]; g = 0.3
xs = []
for w in ws: xs.append(x); x += w + g
panel(1, "Quadros do vídeo", xs[0], yt, ws[0], hh, mosaico)
panel(2, "Características SIFT", xs[1], yt, ws[1], hh, sift_kp)
panel(3, "Correspondências", xs[2], yt, ws[2], hh, corresp)
panel(4, "Poses das câmeras", xs[3], yt, ws[3], hh, poses)
for i in range(3): arrow((xs[i] + ws[i] + 0.02, yt + hh / 2 - 0.05), (xs[i + 1] - 0.02, yt + hh / 2 - 0.05))
# fileira de baixo (direita para esquerda)
w5 = 2.0; x5 = xs[3] + ws[3] - w5
w6 = 1.5; x6 = x5 - g - w6
w7 = x6 - g - 0.04; x7 = 0.04
panel(5, "Triangulação", x5, y2, w5, h2, triang)
panel(6, "Nuvem densa", x6, y2, w6, h2, densa)
panel(7, "Malha final", x7, y2, w7, h2, malha)
xa = xs[3] + ws[3] / 2; arrow((xa, yt - 0.02), (xa, y2 + h2 + 0.02))
arrow((x5 - 0.02, y2 + h2 / 2), (x6 + w6 + 0.02, y2 + h2 / 2))
arrow((x6 - 0.02, y2 + h2 / 2), (x7 + w7 + 0.02, y2 + h2 / 2))

# PDF: matplotlib embute as fotos sem perda (Flate); o Ghostscript recomprime só as imagens em JPEG q~90,
# sem reamostrar (texto e vetores continuam vetoriais).
raw = IMG + "_pipeline_raw.pdf"
fig.savefig(raw, dpi=DPI)
subprocess.run(["gs", "-q", "-dSAFER", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5", "-dAutoFilterColorImages=false",
                "-dColorImageFilter=/DCTEncode", "-dDownsampleColorImages=false", "-dEncodeColorImages=true",
                "-o", IMG + "pipeline.pdf", "-c", "<< /ColorImageDict << /QFactor 0.25 /Blend 1 /HSamples [1 1 1 1]"
                " /VSamples [1 1 1 1] >> >> setdistillerparams", "-f", raw], check=True)
os.remove(raw)
fig.savefig(IMG + "pipeline.png", dpi=DPI, pil_kwargs={"optimize": True})
fig.savefig(IMG + "pipeline.jpg", dpi=DPI, pil_kwargs={"quality": 92, "subsampling": 0})
for f in ("pipeline.pdf", "pipeline.png", "pipeline.jpg"):
    print(f, os.path.getsize(IMG + f) // 1024, "KB", Image.open(IMG + f).size if not f.endswith("pdf") else "")
