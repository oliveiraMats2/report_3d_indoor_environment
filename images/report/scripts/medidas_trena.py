"""Mede no modelo os objetos medidos com trena (mesa, micro-ondas e livro) e compara.

Nuvem: PatchMatch limpa (voxel 1 cm, cm) levada ao referencial do piso por planes/T_floor.npy,
em que a terceira coordenada e a altura h sobre o piso.
Alturas: mediana de h na faixa da superficie. Larguras: a partir do centro, bordas onde a
ocupacao da faixa da superficie, em grade de 1 cm, fica abaixo de 50% em dois bins seguidos.
Regioes e bordas foram escolhidas a mao, ja conhecendo os valores da trena: o criterio de borda foi trocado depois
de a primeira versao dar 70,8 cm na largura do micro-ondas, e o fundo dele e a linha densa em x ~ 150 cm vista na nuvem.
Uso: medidas_trena.py  (somente leitura nos dados; escreve medidas_trena.json ao lado da pasta report)."""
import json, os
import numpy as np, cv2

R = "/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha"
OUT = os.path.join(os.path.dirname(__file__), "..", "medidas_trena.json")

# trena (cm), medida em 04/11/2026
TRENA = {
    "mesa_altura": 78, "mesa_largura": 126, "mesa_profundidade": 53,
    "micro_altura": 32, "micro_largura": 59, "micro_profundidade": 43,
    "livro_comprimento": 20,
}

P = np.load(f"{R}/clean/patch_match/points.npy")
T = np.load(f"{R}/planes/T_floor.npy")
Q = P @ T[:3, :3].T + T[:3, 3]
X, Y, H = Q[:, 0], Q[:, 1], Q[:, 2]


def box(x0, x1, y0, y1, h0, h1):
    return (X > x0) & (X < x1) & (Y > y0) & (Y < y1) & (H > h0) & (H < h1)


def borda(m, eixo, faixa, outra, limiar=0.5):
    """Extensao ao longo de `eixo` (0=x, 1=y): bordas a 50% de ocupacao, faixa `outra` no outro eixo."""
    a, b = Q[m, eixo], Q[m, 1 - eixo]
    s = (b > outra[0]) & (b < outra[1])
    ed = np.arange(faixa[0], faixa[1] + 1, 1.0)
    occ = (np.histogram2d(a[s], b[s], bins=[ed, np.arange(outra[0], outra[1] + 2, 2.0)])[0] > 0).mean(1)
    # caminha do bin mais ocupado para fora e para quando dois bins seguidos ficam abaixo do limiar
    k = int(occ.argmax())
    def anda(passo):
        j = k
        while 0 <= j + passo < len(occ) and not (occ[j + passo] < limiar and (not 0 <= j + 2 * passo < len(occ) or occ[j + 2 * passo] < limiar)):
            j += passo
        return j
    return ed[anda(+1) + 1] - ed[anda(-1)]


def pico(m, passo=0.5):
    h = H[m]
    n, e = np.histogram(h, bins=np.arange(h.min(), h.max() + passo, passo))
    k = n.argmax()
    j = (h > e[k] - 1.0) & (h < e[k + 1] + 1.0)
    return float(np.median(h[j]))


med = {}
# mesa: tampo entre x -125..15 e y -30..40 (vista de cima), faixa de 2,5 cm em torno do pico
h_mesa = pico(box(-130, 15, -40, 45, 68, 82))
h_mesa = float(np.median(H[box(-110, 5, -20, 28, h_mesa - 2.5, h_mesa + 2.5)]))
mt = box(-130, 20, -45, 50, h_mesa - 2.5, h_mesa + 2.5)
med["mesa_altura"] = h_mesa
med["mesa_largura"] = borda(mt, 0, (-135, 20), (-15, 25))
med["mesa_profundidade"] = borda(mt, 1, (-40, 45), (-60, -5))

# micro-ondas: tampo menos o tampo do armario em que ele esta
mm = box(100, 165, -8, 66, 112, 125)
h_top = pico(mm)
h_top = float(np.median(H[box(110, 148, 0, 52, h_top - 2.0, h_top + 2.0)]))
h_arm = pico(box(100, 165, -8, 66, 82, 92))
mm = box(100, 165, -15, 70, h_top - 2.0, h_top + 2.0)
med["micro_altura"] = h_top - h_arm
med["micro_largura"] = borda(mm, 1, (-15, 70), (115, 145))
# fundo: a borda traseira do tampo e a linha densa em x ~ 150 cm; a parede fica em x ~ 160 cm
xs = X[mm & (Y > 5) & (Y < 50)]
n, e = np.histogram(xs, bins=np.arange(140, 156, 1.0))
x_fundo = e[n.argmax()] + 0.5
n, e = np.histogram(xs, bins=np.arange(100, 120, 1.0))
x_frente = e[np.argmax(n > 0.5 * n.max())]
med["micro_profundidade"] = x_fundo - x_frente

# livro no chao: pontos acima do piso perto de (74, 7) cm, retangulo de area minima
d = np.hypot(X - 74, Y - 7)
piso = np.median(H[(d > 18) & (d < 30) & (np.abs(H) < 1.5)])
ml = (d < 16) & (H > piso + 0.6) & (H < 3)
w, l = sorted(cv2.minAreaRect(Q[ml, :2].astype(np.float32))[1])
med["livro_comprimento"] = l
med["livro_largura_sem_trena"] = w

res = {}
for k, v in TRENA.items():
    m = med[k]
    res[k] = dict(trena_cm=v, modelo_cm=round(float(m), 1), erro_cm=round(float(abs(m - v)), 2), erro_rel_pct=round(float(100 * abs(m - v) / v), 2))
res["media"] = dict(erro_cm=round(float(np.mean([res[k]["erro_cm"] for k in TRENA])), 2),
                    erro_rel_pct=round(float(np.mean([res[k]["erro_rel_pct"] for k in TRENA])), 2))
res["livro_largura_sem_trena_cm"] = round(float(med["livro_largura_sem_trena"]), 1)
res["alturas_referencia_cm"] = dict(tampo_mesa=round(h_mesa, 1), tampo_micro=round(h_top, 1), tampo_armario=round(h_arm, 1), piso_livro=round(float(piso), 2))
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
for k, v in res.items():
    print(k, v)
