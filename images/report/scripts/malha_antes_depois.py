"""Malha de Poisson antes e depois do corte por densidade (fig:limpeza, painéis G9).
Rodar: .venv/bin/python malha_antes_depois.py   (a partir de qualquer pasta; o .venv fica em 3d_indoor_environment)
Entradas (somente leitura), em 3d_indoor_environment/cozinha/results/config_cozinha/:
  report_filtragem/poisson_q0.00.ply (antes) e poisson_q{Q}.ply (depois), planes/floor.npy.
Saídas: images/report/malha_antes.jpg (faces removidas em vermelho) e malha_depois.jpg."""
import os, sys, numpy as np, open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from open3d.visualization import rendering as rd
Q = sys.argv[1] if len(sys.argv) > 1 else "0.02"
CORTE = float(sys.argv[2]) if len(sys.argv) > 2 else 255.0   # cm acima do chão: retira o teto
D = "/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha/"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 1000, 820

def carrega(q):
    m = o3d.io.read_triangle_mesh(D + f"report_filtragem/poisson_q{q}.ply")
    return m, np.asarray(m.vertices), np.asarray(m.triangles)

m0, v0, f0 = carrega("0.00")
m1, v1, f1 = carrega(Q)
pl = np.load(D + "planes/floor.npy"); up = pl[:3] / np.linalg.norm(pl[:3])
alt = v0 @ pl[:3] + pl[3]                      # cm acima do chão (floor.npy, up = -y)
# vértices mantidos = vértices de "depois" (mesma ordem, coordenadas idênticas)
vv = lambda a: np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * 3))).ravel()
keep = np.isin(vv(v0), vv(v1))
assert keep.sum() == len(v1)
cut = alt[f0].mean(1) > CORTE                  # faces do teto/acima: fora de ambas as vistas
rem = ~keep[f0].all(1)                         # faces que o corte remove
print("removidos: vértices", (~keep).sum(), "faces", rem.sum(), "(dif. faces", len(f0) - len(f1), ")")

def sub(v, f, c, mask):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f[mask]))
    m.vertex_colors = o3d.utility.Vector3dVector(c)
    m.remove_unreferenced_vertices(); m.compute_vertex_normals(); return m
c0 = np.asarray(m0.vertex_colors)
# vista comum: câmera acima, olhando o centro da cena
sel = alt[f0].mean(1) <= CORTE
ctr = v0[np.unique(f0[sel])].mean(0)
base = np.cross(up, [1, 0, 0]); base /= np.linalg.norm(base); side = np.cross(base, up)
AZ, EL, DIST = float(os.environ.get("AZ", 0)), float(os.environ.get("EL", 55)), float(os.environ.get("DIST", 640))
a, e = np.radians(AZ), np.radians(EL)
hor = np.cos(a) * base + np.sin(a) * side
eye = ctr + DIST * (np.cos(e) * hor + np.sin(e) * up)

def render(mesh, arquivo):
    def quadro(bg):      # renderer novo por fundo; o 1o render sai preto, entao renderiza duas vezes
        r = rd.OffscreenRenderer(W, H)
        mat = rd.MaterialRecord(); mat.shader = os.environ.get("SH", "defaultLit")
        r.scene.add_geometry("m", mesh, mat)
        r.scene.show_skybox(False); r.scene.set_background(bg)
        r.scene.set_lighting(r.scene.LightingProfile.NO_SHADOWS, [0.3, -0.5, -0.8])
        r.setup_camera(55, ctr, eye, up)    # up = normal do chao (aprox. -y)
        r.render_to_image()
        im = np.array(r.render_to_image()).astype(int); del r
        return im
    preto, branco = quadro([0, 0, 0, 1]), quadro([1, 1, 1, 1])
    img = preto.copy()
    assert (np.abs(branco - preto).sum(-1) > 300).mean() > 0.2
    fundo = np.abs(branco - preto).sum(-1) > 300          # pixels onde o fundo aparece
    img[fundo] = 255; img = img.astype(np.uint8)
    return Image.fromarray(img)

def legenda(im, titulo, linha2):
    return im      # sem titulo embutido: o rotulo fica na subcaption do LaTeX (titulo/linha2 so documentam)

fmt = lambda n: f"{n:,}".replace(",", ".")
# antes: faces removidas em vermelho sólido
m_a = sub(v0, f0, np.where(keep[:, None], c0, [0.9, 0.05, 0.05]), ~cut)
# vértices vermelhos pintam faces com algum vértice vermelho; faces 'rem' são exatamente essas ou mais; ok
m_d = sub(v1, f1, np.asarray(m1.vertex_colors), alt_d := (np.asarray(v1) @ pl[:3] + pl[3])[f1].mean(1) <= CORTE)
ia = legenda(render(m_a, "a").convert("RGB"), "Antes", f"{fmt(len(v0))} vértices, {fmt(len(f0))} faces")
id_ = legenda(render(m_d, "d").convert("RGB"), "Depois", f"{fmt(len(v1))} vértices, {fmt(len(f1))} faces")
for im, n in ((ia, "malha_antes"), (id_, "malha_depois")):
    for qual in range(88, 40, -6):
        p = f"{os.environ.get("OUTDIR", OUT)}/{n}.jpg"; im.save(p, quality=qual, optimize=True)
        if os.path.getsize(p) <= 400_000: break
    print(p, os.path.getsize(p))
    a = np.array(im.convert("L")); ys, xs = np.where(a < 250); pad = 20      # recorte ao conteudo
    im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, W), min(ys.max() + pad, H))).save(
        f"{os.environ.get('OUTDIR', OUT)}/{n}_crop.jpg", quality=qual, optimize=True)
