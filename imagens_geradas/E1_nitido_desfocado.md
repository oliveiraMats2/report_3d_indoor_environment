# E1 nitido x desfocado (cozinha, extra, fora do relatorio)

Figura: imagens_geradas/E1_nitido_desfocado.png (220 dpi). Script: scripts/E1_nitido_desfocado.py.

Fonte: 3d_indoor_environment/cozinha/artifacts/config_cozinha/frame_quality_metrics.csv (coluna laplacian_var, calculada a 640 px de largura, src/frame_quality.py); quadros em cozinha/frames/all_stride10/.

Criterio: blur_rel = laplacian_var < 0,5 x mediana. Mediana do CSV = 287,59, limiar = 143,79 (config_cozinha.yaml, frame_quality.blur.relative_to_median 0,5).

| Quadro | laplacian_var | keep | motivo | FAST (CSV) | SIFT (960x540) |
|---|---|---|---|---|---|
| img_2091 (nitido) | 636,84 | True | vazio | 1627 | 3218 |
| img_2090 (desfocado) | 120,91 | False | blur_rel | 329 | 731 |

Quadros consecutivos (passo 10), mesma cesta. Razao nitido/desfocado = 5,27. Recorte 400x300 px nativos da textura da cesta.
