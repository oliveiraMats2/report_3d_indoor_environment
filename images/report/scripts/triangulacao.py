"""Figura fig:tri. Uso: /home/moliveira/Documents/3d_indoor_environment/.venv/bin/python triangulacao.py
Le o modelo COLMAP da cozinha (somente leitura) e desenha a triangulacao do ponto 3046041
(quina do livro sobre a jarra, bancada da pia) nos quadros img_1150 e img_1154. Saida: images/report/triangulacao.jpg"""
import os
import numpy as np, cv2, pycolmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SRC = "/home/moliveira/Documents/3d_indoor_environment/cozinha"
RUN = SRC + "/results/config_cozinha"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "triangulacao.jpg")
PID, IDS = 3046041, (1151, 1155)
NOMES = ("img_1150", "img_1154")
CW, CH = 800, 600  # recorte (px) em torno do keypoint; coordenadas anotadas sao do quadro inteiro

rec = pycolmap.Reconstruction(RUN + "/exports/colmap/sparse/0")
s = float(np.load(RUN + "/scale/scale.npy")[0])  # cm por unidade do modelo (29,3745)
cam = rec.cameras[1]
fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
p = rec.points3D[PID]
X = p.xyz * s
d = {}
for i, n in zip(IDS, NOMES):
    im = rec.images[i]
    assert im.name == n + ".png"
    T = im.cam_from_world()
    e = [q for q in p.track.elements if q.image_id == i][0]
    ob = np.array(im.points2D[e.point2D_idx].xy)
    pr = np.array(cam.img_from_cam(T * p.xyz))
    d[i] = dict(img=cv2.cvtColor(cv2.imread(f"{SRC}/frames/all_stride10/{n}.png"), cv2.COLOR_BGR2RGB),
                R=T.rotation.matrix(), t=T.translation * s, C=im.projection_center() * s,
                ob=ob, err=float(np.linalg.norm(pr - ob)))
base = float(np.linalg.norm(d[IDS[0]]["C"] - d[IDS[1]]["C"]))
print("ponto", PID, "XYZ cm", X, "erro medio", p.error, "track", len(p.track.elements))
for i in IDS: print(i, d[i]["ob"], d[i]["err"], d[i]["C"], d[i]["t"])
print("baseline cm", base)

def f(v, n=1):  # numero com virgula decimal
    return f"{v:.{n}f}".replace(".", ",").replace("-", "−")

W, H = 1920, 1080
FH = 3.6
iw, ih = 1.9, 1.9 * CH / CW
fig = plt.figure(figsize=(10, FH), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, FH); ax.axis("off")
cor = {IDS[0]: "#1f77b4", IDS[1]: "#d95f02"}
Xp = np.array([5.0, 2.85])
x0s = {IDS[0]: 0.05, IDS[1]: 10 - 0.05 - iw}
y0 = 0.42
for i, n in zip(IDS, NOMES):
    ox, oy = d[i]["ob"]
    cx0 = int(np.clip(round(ox - CW / 2), 0, W - CW)); cy0 = int(np.clip(round(oy - CH / 2), 0, H - CH))
    crop = d[i]["img"][cy0:cy0 + CH, cx0:cx0 + CW]
    x0 = x0s[i]
    ax.imshow(crop, extent=[x0, x0 + iw, y0, y0 + ih], zorder=1, aspect="auto")
    ax.add_patch(Rectangle((x0, y0), iw, ih, fill=False, ec=cor[i], lw=2.2, zorder=3))
    kp = np.array([x0 + (ox - cx0) / CW * iw, y0 + (1 - (oy - cy0) / CH) * ih])
    u = (kp - Xp) / np.linalg.norm(kp - Xp)
    Cc = kp + 0.5 * u
    ax.plot([Xp[0], Cc[0]], [Xp[1], Cc[1]], color=cor[i], lw=1.8, zorder=2)
    ax.plot(*kp, "o", mfc="none", mec="#ffeb3b", mew=2, ms=11, zorder=4)
    ax.plot(*Cc, "^", color=cor[i], ms=10, zorder=5)
    ax.text(Cc[0] + 0.1, Cc[1] - 0.02, "C₁" if i == IDS[0] else "C₂", ha="left", va="center", fontsize=10.5,
            color="white", weight="bold", zorder=6, bbox=dict(fc=cor[i], ec="none", pad=1.2))
    d[i]["Cc"] = Cc
    ha = "left" if i == IDS[0] else "right"
    ax.text(x0 if i == IDS[0] else x0 + iw, y0 + ih + 0.05,
            f"{n} (id {i})\nkeypoint ({f(ox)}; {f(oy)}) px\nerro de reprojeção {f(d[i]['err'],2)} px",
            ha=ha, va="bottom", fontsize=10.5, color=cor[i], weight="bold", linespacing=1.15)
ax.plot(*Xp, "o", color="#c62828", ms=13, zorder=6)
ax.text(Xp[0], Xp[1] + 0.15, f"Ponto 3D {PID} (quina do livro)\nX = ({f(X[0])}; {f(X[1])}; {f(X[2])}) cm",
        ha="center", va="bottom", fontsize=11, color="#c62828", weight="bold", linespacing=1.15)
ax.text(Xp[0] + 0.2, Xp[1] - 0.02, f"{len(p.track.elements)} imagens, erro médio {f(p.error,2)} px",
        ha="left", va="center", fontsize=10, color="#c62828")
a_, b_ = d[IDS[0]]["Cc"], d[IDS[1]]["Cc"]
ym = 0.17
ax.annotate("", (b_[0], ym), (a_[0], ym), arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
ax.plot([a_[0], a_[0]], [a_[1] - 0.08, ym], "k:", lw=.9); ax.plot([b_[0], b_[0]], [b_[1] - 0.08, ym], "k:", lw=.9)
ax.text(5, ym, f"linha de base {f(base)} cm", ha="center", va="center", fontsize=10.5,
        bbox=dict(fc="white", ec="none", pad=1.5))
ax.text(9.95, 3.55, "esquema fora de escala", ha="right", va="top", fontsize=9, color="#555")

def bloco(i):
    R, t = d[i]["R"], d[i]["t"]
    lin = [f"[{f(r[0],2):>5} {f(r[1],2):>5} {f(r[2],2):>5}]" for r in R]
    return ("[R | t]\n" + "\n".join(lin) + f"\nt = ({f(t[0])}; {f(t[1])};\n     {f(t[2])}) cm")
for i, x in ((IDS[0], 2.95), (IDS[1], 7.05)):
    ax.text(x, 1.12, bloco(i), ha="center", va="center", fontsize=10, family="DejaVu Sans Mono", multialignment="left",
            bbox=dict(fc="white", ec=cor[i], lw=2, boxstyle="round,pad=0.3"))
ax.text(5, 1.12, "K (comum)\n"
        f"fx = {f(fx)}\nfy = {f(fy)}\ncx = {f(cx,0)}, cy = {f(cy,0)}\n"
        f"k1 = {f(k1,3)}\nk2 = {f(k2,3)}",
        ha="center", va="center", fontsize=10, multialignment="left", linespacing=1.2,
        bbox=dict(fc="#eef3fb", ec="#555", boxstyle="round,pad=0.3"))
fig.savefig(OUT, dpi=200, format="jpg", pil_kwargs=dict(quality=80, optimize=True))
print(os.path.getsize(OUT) / 1024, "KB")
