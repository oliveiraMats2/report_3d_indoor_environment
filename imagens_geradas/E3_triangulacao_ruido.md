# E3 triangulacao com ruido
Script: scripts/E3_triangulacao_ruido.py. Dados: cozinha/results/config_cozinha (sparse/0, sfm/point_error.npy).
Ponto 3D id 2744776 (erro 0,578 px, prof. ~99 cm, track >= 8). Ruido gaussiano por eixo sigma=0,451 px (residuo medio 0,565 px = media de point_error.npy), N=500, cv2.triangulatePoints com poses reais.
Pares: img_1150/img_1151 (baseline 2,82 cm): sigma_z 1,49 cm; img_1150/img_1154 (12,03 cm): sigma_z 0,31 cm. Curva: img_1150 com 1151..1170.
