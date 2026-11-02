"""Fig. fig:limpeza (paineis a e b), recomputacao ilustrativa (std_ratio=2,0, nb=20;
a rodada real usou std_ratio=100, 5 pontos).
(a) esparsa vista de cima inclinada (70 graus), sem teto, com o frustum de img_0127;
(b) quadro real img_0127 com os pontos 3D observados por essa camera (sfm/obs.npy).
Uso: python limpeza_outliers.py [saida]"""
import sys, numpy as np, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
B = "/home/moliveira/Documents/3d_indoor_environment"
R0 = f"{B}/cozinha/results/config_cozinha"
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/moliveira/Documents/report_3d_indoor_environment/images/report/limpeza_outliers_img_0127.jpg"
I = 127                                    # img_0127 (id COLMAP 128)
back, elev, zoom, W, H = 600.0, 70.0, 1150.0, 1000, 697
P = np.load(f"{R0}/scale/points3d_cm.npy")
mask = np.load(f"{R0}/report_filtragem/outlier_mask_std2.0.npy")
fl = np.load(f"{R0}/planes/floor.npy")
K = np.load(f"{R0}/sfm/K.npy"); Rc = np.load(f"{R0}/sfm/cam_R.npy")[I]
C = np.load(f"{R0}/scale/centers_cm.npy")[I]
obs = np.load(f"{R0}/sfm/obs.npy"); obs = obs[obs[:, 1] == I]
pid = obs[:, 0].astype(int); uv = obs[:, 2:4]; mo = mask[pid]
h = P @ fl[:3] + fl[3]
keep = h <= 270                            # sem teto
Pk, mk = P[keep], mask[keep]
up = fl[:3] / np.linalg.norm(fl[:3])
tgt = np.median(Pk[~mk], axis=0)
d = C - tgt; d -= (d @ up) * up; d /= np.linalg.norm(d)
tgt = tgt + 60 * d                         # desloca a vista para incluir a camera
el = np.radians(elev)
eye = tgt + back * (np.cos(el) * d + np.sin(el) * up)
z = (tgt - eye) / np.linalg.norm(tgt - eye)
x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x)
Mv = np.stack([x, y, z])
def proj(Q):
    Xc = (Q - eye) @ Mv.T
    return zoom * Xc[:, 0] / Xc[:, 2] + W / 2, zoom * Xc[:, 1] / Xc[:, 2] + H / 2, Xc[:, 2]
u, v, dz = proj(Pk)
ok = dz > 20; u, v, dz, m = u[ok], v[ok], dz[ok], mk[ok]
o = np.argsort(-dz); u, v, m = u[o], v[o], m[o]

n, n_out = len(mask), int(mask.sum()); n_in = n - n_out
br = lambda k: f"{k:,}".replace(",", ".")
pc = lambda a, b: f"{100*a/b:.2f}".replace(".", ",")
GR, RD, BL = "#8a8a8a", "#d62728", "#1f5fbf"

fig = plt.figure(figsize=(6.7, 2.3), dpi=200)
a = fig.add_axes([0.0, 0.0, 0.493, 1.0]); a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
a.scatter(u[~m], v[~m], s=0.4, c=GR, lw=0, rasterized=True)
a.scatter(u[m], v[m], s=4, c=RD, lw=0, rasterized=True)
# frustum de img_0127 (alcance 120 cm) e eixo optico
cs = np.array([[0, 0], [1920, 0], [1920, 1080], [0, 1080]], float)
rays = np.c_[(cs[:, 0] - K[0, 2]) / K[0, 0], (cs[:, 1] - K[1, 2]) / K[1, 1], np.ones(4)]
cor = C + 80 * (rays @ Rc)               # canto no mundo (R^T r)
pu, pv, _ = proj(np.vstack([C, cor, C + 150 * Rc[2]]))
for k in range(4):
    a.plot([pu[0], pu[k + 1]], [pv[0], pv[k + 1]], c=BL, lw=1.0)
    a.plot([pu[k + 1], pu[(k + 1) % 4 + 1]], [pv[k + 1], pv[(k + 1) % 4 + 1]], c=BL, lw=1.0)
a.annotate("", xy=(pu[5], pv[5]), xytext=(pu[0], pv[0]),
           arrowprops=dict(arrowstyle="-|>", color=BL, lw=1.4))
a.plot(pu[0], pv[0], "o", c=BL, ms=4)
a.text(pu[0] + 12, pv[0] + 28, "img_0127", color=BL, fontsize=11, weight="bold",
       bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.8))
a.text(12, 14, "(a) Esparsa, vista de cima\n20 vizinhos, desvio 2,0",
       fontsize=11.5, va="top", linespacing=1.35, bbox=dict(fc="white", ec="#444", lw=0.5, pad=2.5))
a.scatter([], [], s=12, c=GR, label="mantidos"); a.scatter([], [], s=12, c=RD, label="removidos")
a.legend(loc="lower right", fontsize=11.5, frameon=True, facecolor="white", handletextpad=0.2, borderpad=0.4)

b = fig.add_axes([0.507, 0.5 - 0.5 * (0.493 * 6.7 * 1080 / 1920) / 2.3, 0.493, (0.493 * 6.7 * 1080 / 1920) / 2.3])
img = cv2.cvtColor(cv2.imread(f"{B}/cozinha/frames/all_stride10/img_0127.png"), cv2.COLOR_BGR2RGB)
b.imshow(cv2.addWeighted(img, 0.6, np.full_like(img, 255), 0.4, 0))
b.scatter(uv[~mo, 0], uv[~mo, 1], s=0.7, c="#555555", lw=0, rasterized=True)
b.scatter(uv[mo, 0], uv[mo, 1], s=6, c=RD, lw=0, rasterized=True)
b.set_xlim(0, 1920); b.set_ylim(1080, 0); b.axis("off")
no, nr = len(pid), int(mo.sum())
b.text(25, 25, "(b) Quadro img_0127\n20 vizinhos, desvio 2,0",
       fontsize=11.5, va="top", linespacing=1.35, bbox=dict(fc="white", ec="#444", lw=0.5, pad=2.5))
fig.savefig(OUT, dpi=200, pil_kwargs=dict(quality=82, optimize=True))
print(OUT, n_in, n_out, n, pc(n_out, n), no, nr, pc(nr, no))
