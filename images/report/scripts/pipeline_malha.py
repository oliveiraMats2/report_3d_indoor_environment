"""Renders 3D em alta resolução para os painéis 6 (nuvem densa) e 7 (malha final) de fig:pipeline.
Rodar uma vez antes de pipeline.py:
  /home/moliveira/Documents/3d_indoor_environment/.venv/bin/python pipeline_malha.py
Saídas (cache, neste diretório):
  _densa_hi.png  nuvem PatchMatch limpa, mesma câmera/recorte de densa.py (G12), câmeras em azul;
  _malha_hi.png  malha limpa final (malha_limpa/malha_limpa.ply) sem teto, sombreada na mão.
Ambos renderizados com supersampling SS x (point_size e line_width multiplicados por SS);
a redução LANCZOS até o tamanho do painel é feita em pipeline.py.
Somente leitura em /home/moliveira/Documents/3d_indoor_environment."""
import os, sys, runpy, tempfile, numpy as np, open3d as o3d
from PIL import Image
from open3d.visualization import rendering as rd

R = "/home/moliveira/Documents/3d_indoor_environment"
D = R + "/cozinha/results/config_cozinha/"
HERE = os.path.dirname(os.path.abspath(__file__))
SS = 2   # supersampling em relação aos renders originais (densa 1700x1150, malha 1000x800)
QUAIS = sys.argv[1:] or ["densa", "malha"]


def render2(r):
    """O OffscreenRenderer devolve quadro preto no 1o render: renderiza duas vezes."""
    r.render_to_image()
    return np.asarray(r.render_to_image()).copy()


def autocrop(im, pad=20 * SS):
    """Recorta ao conteúdo (fundo = cor do canto), como densa.py."""
    m = np.abs(im.astype(int) - im[5, 5].astype(int)).sum(2) > 30; m[-3:] = False
    ys, xs = np.where(m)
    return im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


if "densa" in QUAIS:
    # reaproveita densa.py inteiro (filtro de espúrios, referencial da sala, câmeras), sem o bloco __main__;
    # o json de contagens vai para um diretório temporário para não tocar em images/report.
    tmp = tempfile.mkdtemp()
    argv = sys.argv; sys.argv = ["densa.py", R, tmp]
    g = runpy.run_path(os.path.join(HERE, "densa.py"), run_name="densa_lib")
    sys.argv = argv
    q, col, esp = g["q"], g["col"], g["esp"]
    v = (~esp) & (q[:, 2] < g["H_CORTE"]) & (q[:, 1] > g["Y_JANELA"])
    W, H = 1700 * SS, 1150 * SS
    r, ctr = g["render"](q[v], col[v], [(g["cameras_lineset"](), 3.0 * SS)], None, W, H, 3.0 * SS)
    r.scene.camera.set_projection(40, W / H, 10, 5000, rd.Camera.FovType.Vertical)
    r.scene.camera.look_at(ctr, ctr + np.array((40, -430, 330)), [0, 0, 1])
    im = autocrop(render2(r)[..., :3])
    Image.fromarray(im).save(os.path.join(HERE, "_densa_hi.png"))
    print("densa", im.shape)
    del r

if "malha" in QUAIS:
    ml = o3d.io.read_triangle_mesh(D + "malha_limpa/malha_limpa.ply")   # malha limpa final (6888396 faces)
    v = np.asarray(ml.vertices); f = np.asarray(ml.triangles); c = np.asarray(ml.vertex_colors)
    pl = np.load(D + "planes/floor.npy"); up = pl[:3] / np.linalg.norm(pl[:3])
    alt = v @ pl[:3] + pl[3]
    keep = alt[f].mean(1) <= float(os.environ.get("CORTE", 255))
    idx = np.unique(f[keep])                      # vértices usados, na ordem que remove_unreferenced_vertices mantém
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f[keep]))
    m.remove_unreferenced_vertices(); m.compute_vertex_normals()
    n = np.asarray(m.vertex_normals); vv = np.asarray(m.vertices); col = c[idx]
    ctr = vv.mean(0)
    base = np.cross(up, [1, 0, 0]); base /= np.linalg.norm(base); side = np.cross(base, up)
    AZ, EL, DIST = float(os.environ.get("AZ", 0)), float(os.environ.get("EL", 55)), float(os.environ.get("DIST", 640))
    a, e = np.radians(AZ), np.radians(EL)
    eye = ctr + DIST * (np.cos(e) * (np.cos(a) * base + np.sin(a) * side) + np.sin(e) * up)
    d = eye - vv; d /= np.linalg.norm(d, axis=1, keepdims=True)
    sh = 0.45 + 0.55 * np.abs((n * d).sum(1))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(col * sh[:, None], 0, 1))
    W, H = 1000 * SS, 800 * SS

    def shot(bg):
        r = rd.OffscreenRenderer(W, H); mat = rd.MaterialRecord(); mat.shader = "defaultUnlit"
        r.scene.add_geometry("m", m, mat); r.scene.show_skybox(False); r.scene.set_background(bg)
        r.setup_camera(55, ctr, eye, up)
        im = render2(r)[..., :3].astype(int); del r
        return im
    p, b = shot([0, 0, 0, 1]), shot([1, 1, 1, 1])
    p[np.abs(b - p).sum(-1) > 300] = 255           # fundo: pixels que mudam com a cor de fundo
    im = autocrop(p.astype(np.uint8))
    Image.fromarray(im).save(os.path.join(HERE, "_malha_hi.png"))
    print("malha", im.shape)
