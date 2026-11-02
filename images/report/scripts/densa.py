#!/usr/bin/env python
"""Gera densa_patch_match.jpg (G12) e densa_espurios.jpg (G13) e imprime as contagens.
Uso: R/.venv/bin/python densa.py [R_REPO] [saida_dir] [--so-g13]
Entradas (cm): clean/patch_match/points.npy e colors.npy; mesh/room_fill/{T_room,footprint_cm}.npy;
scale/centers_cm.npy; sfm/cam_R.npy. Referencial da sala: p_sala = inv(T_room) p -> (x, y, h) com h = altura sobre o piso."""
import sys, json, numpy as np, open3d as o3d
from pathlib import Path
from PIL import Image
from scipy.spatial import cKDTree
from matplotlib.path import Path as MPath

ARGS = [x for x in sys.argv[1:] if not x.startswith("--")]   # --so-g13: gera só densa_espurios.jpg
R = Path(ARGS[0] if len(ARGS) > 0 else "/home/moliveira/Documents/3d_indoor_environment")
OUT = Path(ARGS[1] if len(ARGS) > 1 else Path(__file__).resolve().parent.parent)
RUN = R / "cozinha/results/config_cozinha"
MARGEM = 20.0      # cm fora do contorno da sala tolerados (espessura de parede e erro de plano)
H_PISO = -10.0     # cm abaixo do piso
H_TETO = 300.0     # cm (teto medido ~289,5 cm + 10)
K_VIZ, K_STD = 20, 2.0     # remoção estatística do repositório: 20 vizinhos, 2 desvios
PS_G13 = (3.0, 1.5)   # tamanho do ponto em G13 (mantidos, espúrios), em px da imagem final
H_CORTE = 270.0    # corte do teto na visualização
Y_JANELA = -225.0  # corta a parede/entorno do lado da janela (y < -225) para olhar para dentro

p = np.load(RUN / "clean/patch_match/points.npy"); col = np.load(RUN / "clean/patch_match/colors.npy")
if col.max() > 1.5: col = col / 255.0
Ti = np.linalg.inv(np.load(RUN / "mesh/room_fill/T_room.npy"))
fp = np.load(RUN / "mesh/room_fill/footprint_cm.npy")
q = p @ Ti[:3, :3].T + Ti[:3, 3]
N = len(q)

# --- critério de espúrio (reprodutível) ---
poly = MPath(np.vstack([fp, fp[:1]]))
dentro = poly.contains_points(q[:, :2])
# distância horizontal ao contorno (0 se dentro)
def dist_poly(xy):
    d = np.full(len(xy), np.inf)
    for a, b in zip(fp, np.roll(fp, -1, 0)):
        ab = b - a; t = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(xy - (a + t[:, None] * ab), axis=1))
    return np.where(dentro, 0.0, d)
d_out = dist_poly(q[:, :2])
c_fora = d_out > MARGEM
c_alt = (q[:, 2] < H_PISO) | (q[:, 2] > H_TETO)
# flutuantes/ruído: distância média aos K vizinhos acima de média + K_STD desvios (dentro do que sobra)
base = ~(c_fora | c_alt)
tree = cKDTree(q)
dk = tree.query(q, k=K_VIZ + 1, workers=-1)[0][:, 1:].mean(1)
lim = dk[base].mean() + K_STD * dk[base].std()
c_flut = base & (dk > lim)
esp = c_fora | c_alt | c_flut
cont = dict(total=N, fora_contorno=int(c_fora.sum()), fora_altura=int((c_alt & ~c_fora).sum()),
            flutuante_ruido=int(c_flut.sum()), espurios=int(esp.sum()), pct=round(100 * esp.mean(), 2),
            limiar_dk_cm=round(float(lim), 3), margem_cm=MARGEM,
            fora_contorno_pct=round(100 * c_fora.mean(), 2), flut_pct=round(100 * c_flut.mean(), 2))
print(json.dumps(cont, indent=1))
(OUT / "densa_contagens.json").write_text(json.dumps(cont, indent=1))

# --- câmeras ---
C = np.load(RUN / "scale/centers_cm.npy"); Rc = np.load(RUN / "sfm/cam_R.npy")
cs = C @ Ti[:3, :3].T + Ti[:3, 3]
fw = np.einsum("ij,nj->ni", Ti[:3, :3], Rc[:, 2, :])      # direção de visada no referencial da sala
rt = np.einsum("ij,nj->ni", Ti[:3, :3], Rc[:, 0, :]); dn = np.einsum("ij,nj->ni", Ti[:3, :3], Rc[:, 1, :])
sel = np.arange(0, len(cs), 40)

def cameras_lineset(cor=(0.0, 0.25, 1.0), L=32.0):
    pts, lines = [], []
    hw, hh = 0.22 * L * 0.5, 0.22 * L * 0.28
    for i in sel:
        o = cs[i]; f = o + L * fw[i]
        cantos = [f + s * hw * rt[i] + t * hh * dn[i] for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        base_i = len(pts); pts += [o] + cantos
        lines += [[base_i, base_i + k] for k in (1, 2, 3, 4)] + [[base_i + k, base_i + 1 + k % 4] for k in (1, 2, 3, 4)]
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(pts)), o3d.utility.Vector2iVector(np.array(lines)))
    ls.paint_uniform_color(cor); return ls

def traj_lineset():
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(cs), o3d.utility.Vector2iVector(np.c_[np.arange(len(cs) - 1), np.arange(1, len(cs))]))
    ls.paint_uniform_color((0.15, 0.35, 0.95)); return ls

def render(P, C_, lines, fn, W=1600, H=1100, psize=2.0, eye_dir=(0.0, -1.0, 0.0)):
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    sc = r.scene; sc.set_background([1, 1, 1, 1])
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P); pc.colors = o3d.utility.Vector3dVector(C_)
    m = o3d.visualization.rendering.MaterialRecord(); m.shader = "defaultUnlit"; m.point_size = psize
    sc.add_geometry("pc", pc, m)
    for k, (ls, w) in enumerate(lines):
        ml = o3d.visualization.rendering.MaterialRecord(); ml.shader = "unlitLine"; ml.line_width = w
        sc.add_geometry(f"l{k}", ls, ml)
    ctr = np.array([(fp[:, 0].min() + fp[:, 0].max()) / 2, (fp[:, 1].min() + fp[:, 1].max()) / 2, 110.0])
    return r, ctr

def foto(P, C_, lines, fn, eye_off, up=(0, 0, 1), W=1700, H=1150, psize=2.5, fov=38):
    r, ctr = render(P, C_, lines, fn, W, H, psize)
    r.scene.camera.set_projection(fov, W / H, 10, 5000, o3d.visualization.rendering.Camera.FovType.Vertical)
    r.scene.camera.look_at(ctr, ctr + np.array(eye_off), list(up))
    r.render_to_image(); img = np.array(r.render_to_image()); Image.fromarray(img).save(fn); del r
    return img

# --- G13: vista quase zenital, mantidos em verde e espúrios em vermelho ---
VERDE_PISO = np.array([0.08, 0.40, 0.08])    # mantidos: verde escuro no piso ...
VERDE_TOPO = np.array([0.45, 0.80, 0.40])    # ... clareando até o topo das paredes (dá forma à sala)
VERMELHO = np.array([0xd6, 0x27, 0x28]) / 255.0   # espúrios (#d62728)
MOLDURA = (0.50, 0.05, 0.20, 0.14)   # margens do quadro (esq., dir., cima, baixo) em fração da sala; a faixa da janela fica à esquerda
SS = 2                                      # supersampling: renderiza 2x e reduz com LANCZOS
LARG_FINAL = 1400                           # px (painel de ~3,3 pol no PDF)

def g13(fn, psize=PS_G13, por_cima="esp"):
    v2 = (q[:, 2] < H_CORTE) & (q[:, 1] > Y_JANELA - 300) & (q[:, 2] > -60)
    t = np.clip(q[:, 2] / H_CORTE, 0, 1)[:, None]
    cc = VERDE_PISO * (1 - t) + VERDE_TOPO * t
    cc[esp] = VERMELHO
    W, H = 1700 * SS, 1150 * SS
    ctr = np.array([(fp[:, 0].min() + fp[:, 0].max()) / 2, (fp[:, 1].min() + fp[:, 1].max()) / 2, 110.0])
    eye = ctr + np.array([0, -450, 1300])
    P = q[v2].copy()
    # a classe "por cima" é puxada em direção ao olho ao longo do próprio raio (p' = olho + 0,85 (p - olho)):
    # a projeção não muda, só a profundidade, então ela é desenhada na frente da outra.
    top = esp[v2] if por_cima == "esp" else ~esp[v2]
    P[top] = eye + 0.85 * (P[top] - eye)
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    sc = r.scene; sc.set_background([1, 1, 1, 1]); sc.view.set_post_processing(False)
    for nome, k, ps in (("mant", ~esp[v2], psize[0]), ("esp", esp[v2], psize[1])):
        pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(P[k]); pc.colors = o3d.utility.Vector3dVector(cc[v2][k])
        m = o3d.visualization.rendering.MaterialRecord(); m.shader = "defaultUnlit"; m.point_size = ps * SS
        sc.add_geometry(nome, pc, m)
    sc.camera.set_projection(40, W / H, 10, 5000, o3d.visualization.rendering.Camera.FovType.Vertical)
    sc.camera.look_at(ctr, eye, [-1, 0, 0])   # "up" -x: mesma orientação de esparsa_topo e planta (chanfro embaixo à direita)
    r.render_to_image(); a = np.asarray(r.render_to_image()).copy(); del r
    # enquadra a sala: caixa dos pixels verdes (mantidos ficam a <= 20 cm do contorno) + margem
    g = (a[..., 1].astype(int) - a[..., 0] > 40) & (a[..., 1].astype(int) - a[..., 2] > 40)
    ys, xs = np.where(g); gw, gh = xs.max() - xs.min(), ys.max() - ys.min()
    # inclui a faixa do exterior visto pela janela; espúrios mais distantes ficam fora do quadro
    x0, x1, y0, y1 = xs.min() - int(MOLDURA[0] * gw), xs.max() + int(MOLDURA[1] * gw), ys.min() - int(MOLDURA[2] * gh), ys.max() + int(MOLDURA[3] * gh)
    im = Image.fromarray(a).crop((max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)))
    im = im.resize((LARG_FINAL, round(im.height * LARG_FINAL / im.width)), Image.LANCZOS)
    # legenda em faixa branca acima: fonte de ~5 pt efetivos (como os rotulos dos outros paineis) com o painel a 0,49\textwidth (3,27 pol)
    from PIL import ImageDraw, ImageFont
    dpi = LARG_FINAL / 3.27; fs = round(5.0 / 72 * dpi)
    fonte = ImageFont.truetype("DejaVuSans.ttf", fs)
    faixa = round(1.25 * fs)
    out = Image.new("RGB", (im.width, im.height + faixa), "white"); out.paste(im, (0, faixa))
    d = ImageDraw.Draw(out); mk = round(0.75 * fs)
    itens = [("mantidos", tuple(int(255 * c) for c in (VERDE_PISO + VERDE_TOPO) / 2)), ("espúrios", tuple(int(255 * c) for c in VERMELHO))]
    larg = sum(mk + 0.35 * fs + d.textlength(txt, font=fonte) for txt, _ in itens) + 1.2 * fs
    x = (out.width - larg) / 2; yc = faixa / 2
    for txt, cor in itens:
        d.rectangle((x, yc - mk / 2, x + mk, yc + mk / 2), fill=cor)
        x += mk + 0.35 * fs
        d.text((x, yc), txt, fill=(30, 30, 30), font=fonte, anchor="lm")
        x += d.textlength(txt, font=fonte) + 1.2 * fs
    out.save(fn, quality=90, optimize=True)
    return out

if __name__ == "__main__":
    if "--so-g13" in sys.argv:
        g13(OUT / "densa_espurios.jpg"); sys.exit()
    # G12: sem teto, sem o lado da janela, só dentro da sala
    v = (~esp) & (q[:, 2] < H_CORTE) & (q[:, 1] > Y_JANELA)
    OFF = (40, -430, 330)
    g12 = OUT / "densa_patch_match.png"
    foto(q[v], col[v], [(cameras_lineset(), 3.0)], g12, OFF, psize=3.0, fov=40)
    for f in (g12,):   # recorta ao conteúdo, converte para JPEG e remove o PNG intermediário
        im = Image.open(f).convert("RGB"); a = np.asarray(im).astype(int)
        m = np.abs(a - a[5, 5]).sum(2) > 30; m[-3:] = False; ys, xs = np.where(m)
        pad = 20; im = im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width), min(ys.max() + pad, im.height)))
        im.thumbnail((1400, 1400)); im.save(f.with_suffix(".jpg"), quality=82, optimize=True); f.unlink()
    g13(OUT / "densa_espurios.jpg")   # G13: todos os pontos (sem teto), mantidos em verde e espúrios em vermelho
