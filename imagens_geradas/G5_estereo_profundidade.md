# G5 Estéreo e profundidade (figura extra, não entra no relatório)

Arquivo: G5_estereo_profundidade.png (3600x2400, 200 dpi). Script: scripts/G5_estereo_profundidade.py.

Dados de origem
- Poses e K: R/cozinha/results/config_cozinha/dense/undistorted/sparse (PINHOLE 1600x896, fx=1313,48, fy=1312,64), mesmas poses de exports/colmap/sparse/0.
- Imagens: dense/undistorted/images/img_1150.png e img_1154.png.
- Profundidade de referência: dense/patch_match/ws/stereo/depth_maps/img_1150.png.geometric.bin, multiplicada por 29,3745 cm/unidade.

Método
- Par img_1150 (esquerda) e img_1154 (direita). R,t relativos das poses; rotação relativa 11,7 graus; B = |t| x 29,3745 = 12,03 cm.
- cv2.stereoRectify (alpha=-1) para as rotações; nova K comum com f = 1313,06 px, ponto principal recentrado.
- StereoSGBM (CLAHE, blockSize 5, faixa de busca 96 a 320 px, mediana 5x5), Z = f*B/d.
- Faixa de busca definida a partir da escala da cena (PatchMatch: 63 a 131 cm, percentis 5 e 95). Z estéreo reprojetado ao quadro original da img_1150 para comparar pixel a pixel.

Números
- f = 1313,06 px; B = 12,03 cm; f*B = 15.800 px*cm.
- Pixels comparados: 146.709. Diferença (estéreo menos PatchMatch): mediana +7,2 cm; |diferença| mediana 31,7 cm; erro relativo mediano 37,0%.

Leitura: o estéreo de um único par com 12 cm de linha de base e SGBM é ruidoso e esparso (cobertura de cerca de 12% da imagem retificada) frente ao PatchMatch multivista; o viés é pequeno, mas a dispersão é grande.
