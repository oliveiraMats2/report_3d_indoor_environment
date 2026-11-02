"""G7: RGB, profundidade PatchMatch e profundidade LiDAR (raycast) na mesma pose.
Uso: profundidade.py [img_XXXX] [--teste]  (somente leitura nos dados)."""
import sys, struct
import numpy as np, cv2, pycolmap, open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/moliveira/Documents/3d_indoor_environment"
RUN = f"{R}/cozinha/results/config_cozinha"
DENSE = f"{RUN}/dense"
T_NPY = "/home/moliveira/Documents/3d_indoor_environment_report/outputs/t5_lidar/run/lidar/T_clean_patch_match.npy"
MESH = f"{R}/cozinha_lidar/fused/mesh_closed_2M.ply"
ESC = 29.3745  # cm por unidade do modelo (fonte: notas/t5_lidar.md / plano P1)
OUT = "/home/moliveira/Documents/report_3d_indoor_environment/images/report/profundidade_rgb_recon_lidar.jpg"
import os; OUT = os.environ.get("OUT", OUT)
VMIN, VMAX = 50, 250  # cm

def read_map(p):
    with open(p, "rb") as f:
        buf = f.read()
    i = [k for k, ch in enumerate(buf) if ch == 38][2] + 1  # 3o '&'
    w, h, c = [int(x) for x in buf[:i - 1].decode().split("&")]
    return np.frombuffer(buf[i:], np.float32).reshape((h, w))

_cache = {}
def setup():
    if _cache: return _cache
    rec = pycolmap.Reconstruction(f"{DENSE}/undistorted/sparse")
    cam = list(rec.cameras.values())[0]
    T = np.load(T_NPY); s = np.cbrt(abs(np.linalg.det(T[:3, :3])))
    m = o3d.io.read_triangle_mesh(MESH)
    m.scale(100.0, center=(0, 0, 0))  # m para cm
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
    _cache.update(rec=rec, cam=cam, T=T, s=s, sc=sc); return _cache

def lidar_depth(name):
    S = setup(); cam = S["cam"]; T = S["T"]; s = S["s"]
    im = next(i for i in S["rec"].images.values() if i.name == name + ".png")
    P = im.cam_from_world().matrix()  # 3x4, unidades do modelo
    Rw, t = P[:, :3], P[:, 3]
    C = -Rw.T @ t * ESC                       # centro em cm do modelo
    Rl = T[:3, :3] / s
    C2 = T[:3, :3] @ C + T[:3, 3]             # centro em cm do LiDAR
    Rc2 = Rw @ Rl.T                           # cam_from_lidar
    fx, fy, cx, cy = cam.params; W, H = cam.width, cam.height
    K = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3d.core.float64)
    E = np.eye(4); E[:3, :3] = Rc2; E[:3, 3] = -Rc2 @ C2
    rays = S["sc"].create_rays_pinhole(K, o3d.core.Tensor(E), W, H)
    th = S["sc"].cast_rays(rays)["t_hit"].numpy()
    d = th / s  # raios com z=1 (Open3D): t_hit ja e a profundidade; cm do modelo (divide pela escala)
    d[~np.isfinite(th)] = np.nan
    return d

def recon_depth(name):
    d = read_map(f"{DENSE}/patch_match/ws/stereo/depth_maps/{name}.png.geometric.bin") * ESC
    d[d <= 0] = np.nan; return d

def stats(name):
    b, c = recon_depth(name), lidar_depth(name)
    ok = np.isfinite(b) & np.isfinite(c)
    return b, c, ok, np.abs(b - c)[ok]

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--teste" in sys.argv:
        for n in args:
            b, c, ok, dd = stats(n)
            print(n, "recon", np.isfinite(b).mean().round(3), "lidar", np.isfinite(c).mean().round(3),
                  "ambos", ok.mean().round(3), "mediana", np.median(dd).round(2) if dd.size else None); 
        sys.exit()
    name = args[0] if args else "img_1740"
    b, c, ok, dd = stats(name)
    rgb = cv2.cvtColor(cv2.imread(f"{DENSE}/undistorted/images/{name}.png"), cv2.COLOR_BGR2RGB)
    diff = np.where(ok, np.abs(b - c), np.nan)
    print(name, "mediana |b-c| (cm):", np.median(dd), "n", ok.sum(), "p90", np.percentile(dd, 90))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.30), gridspec_kw=dict(wspace=0.04, left=.003, right=.885, top=.88, bottom=.02))
    cm = plt.get_cmap("turbo").copy(); cm.set_bad("#222222")
    ax[0].imshow(rgb); ax[0].set_title("(a) quadro RGB", fontsize=7, pad=2)
    ax[1].imshow(np.ma.masked_invalid(b), cmap=cm, vmin=VMIN, vmax=VMAX); ax[1].set_title("(b) reconstrução", fontsize=7, pad=2)
    im = ax[2].imshow(np.ma.masked_invalid(c), cmap=cm, vmin=VMIN, vmax=VMAX); ax[2].set_title("(c) LiDAR", fontsize=7, pad=2)
    cm2 = plt.get_cmap("magma").copy(); cm2.set_bad("#222222")
    im2 = ax[3].imshow(np.ma.masked_invalid(diff), cmap=cm2, vmin=0, vmax=20); ax[3].set_title("(d) |b − c|", fontsize=7, pad=2)
    for a in ax: a.axis("off")
    c1 = fig.add_axes([.895, .10, .008, .70]); cb = fig.colorbar(im, cax=c1); cb.set_label("profundidade (cm)", fontsize=6, labelpad=2)
    c2 = fig.add_axes([.955, .10, .008, .70]); cb2 = fig.colorbar(im2, cax=c2); cb2.set_label("diferença (cm)", fontsize=6, labelpad=2)
    for q in (cb, cb2): q.ax.tick_params(labelsize=6, length=2, pad=1)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", pad_inches=0.02, pil_kwargs=dict(quality=85, optimize=True))
