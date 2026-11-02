"""Figura fig:matches: recorta e reduz as figuras do repo (par img_2299__img_2326, SIFT).
Entrada: R/cozinha/results/config_cozinha/figures/{matching,epilines}/sift/img_2299__img_2326.png
Uso: R/.venv/bin/python matches.py"""
from PIL import Image
import os
R = "/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha/figures"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
W = 1330
m = Image.open(f"{R}/matching/sift/img_2299__img_2326.png").convert("RGB")
h = m.height // 2
m = m.crop((0, 50, m.width, h))  # painel de cima (pos Lowe), sem faixa preta de titulo
m = m.resize((W, round(W * m.height / m.width)), Image.LANCZOS)
m.save(f"{OUT}/matches_img_2299__img_2326.jpg", quality=85, optimize=True)
e = Image.open(f"{R}/epilines/sift/img_2299__img_2326.png").convert("RGB")
e = e.resize((W, round(W * e.height / e.width)), Image.LANCZOS)
e.save(f"{OUT}/epilines_img_2299__img_2326.jpg", quality=85, optimize=True)
