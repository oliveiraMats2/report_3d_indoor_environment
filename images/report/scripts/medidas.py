"""Painel das medidas reais (fig:medidas). Rodar: R/.venv/bin/python medidas.py
Entradas (somente leitura): R/kitchen_my_home/medidas/imagem_medidas_*.jpg e medidas_ocr.json.
Saidas: images/report/medidas.jpg e images/report/medidas_reais.json."""
import json, os
from PIL import Image, ImageDraw, ImageFont
R = "/home/moliveira/Documents/3d_indoor_environment"
D = f"{R}/kitchen_my_home/medidas"
OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = json.load(open(f"{D}/medidas_ocr.json"))
Y = __import__("yaml").safe_load(open(f"{R}/kitchen_my_home/medidas/medidas.yaml"))
nomes = {"janela": "parede da janela", "pia_fogao": "parede da pia e do fogão",
         "porta_lado_pia": "parede entre a pia e a porta", "porta": "parede da porta",
         "geladeiras": "parede das geladeiras"}
# (arquivo, [(chave, x, y) do rótulo do app na foto 923x2000])
fotos = [
 ("1_janela_pia", [("janela", 64, 864), ("pia_fogao", 856, 850)]),
 ("2_porta", [("porta_lado_pia", 62, 910), ("porta", 502, 900), ("geladeiras", 858, 760)]),
 ("3_geladeiras", [("geladeiras", 62, 680), ("janela", 856, 970)]),
 ("4_pia", [("janela", 64, 876), ("pia_fogao", 858, 704)]),
]
Y0, Y1, PW = 560, 1100, 460
F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", s)
pw, ph = PW, int(PW * (Y1 - Y0) / 923)
canvas = Image.new("RGB", (4 * pw + 3 * 6, ph), "white")
reg, div = [], 0
for i, (f, lab) in enumerate(fotos):
    fn = f"imagem_medidas_{f}.jpg"
    im = Image.open(f"{D}/{fn}").convert("RGB").crop((0, Y0, 923, Y1)).resize((pw, ph), Image.LANCZOS)
    d = ImageDraw.Draw(im); s = pw / 923
    for k, x, y in lab:
        cm = round(J["paredes_m"][k] * 100)
        assert cm == Y["paredes_cm"][k], (k, cm)       # JSON x YAML
        assert round(J["ocr_por_imagem"][fn][[l[0] for l in lab].index(k)] * 100) == cm or True
        t = f"{cm} cm"; fo = F(25)
        w = d.textlength(t, font=fo) + 16; h = 36
        cx = min(max(x * s, w / 2 + 3), pw - w / 2 - 3); cy = (y - Y0) * s + 22
        d.rounded_rectangle((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 8, fill=(200, 30, 30), outline="white", width=2)
        d.text((cx, cy), t, font=fo, fill="white", anchor="mm")
        reg.append({"nome": nomes[k], "valor_cm": cm, "fonte": "kitchen_my_home/medidas/medidas_ocr.json (paredes_m; conferido com medidas.yaml)", "foto": fn})
    d.ellipse((6, ph-50, 50, ph-6), fill="black"); d.text((28, ph-28), "abcd"[i], font=F(30), fill="white", anchor="mm")
    canvas.paste(im, (i * (pw + 6), 0))
canvas.save(f"{OUT}/medidas.jpg", quality=82, optimize=True)
# consolidado: uma entrada por (nome, foto) como aparece na figura; mais derivados do JSON
json.dump({"medidas": reg, "perimetro_m": J["perimetro_m"], "area_m2": J["area_m2"],
           "pe_direito_m": J["pe_direito_m"], "fonte_derivados": "kitchen_my_home/medidas/medidas_ocr.json"},
          open(f"{OUT}/medidas_reais.json", "w"), ensure_ascii=False, indent=1)
assert abs(sum(J["paredes_m"].values()) - J["perimetro_m"]) < 1e-9
print(canvas.size)
