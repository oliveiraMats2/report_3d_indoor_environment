# G10 remocao_outliers_2d (figura extra, nao entra no relatorio)
Exemplo: diagramas_exemplo/image-8.png. Script: scripts/G10_remocao_outliers_2d.py (so leitura; usa .venv do repo R).
Dados (R = /home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha):
- clean/patch_match/points.npy, mesh/room_fill/{T_room,footprint_cm}.npy (regra de malha_limpa/ESPECIFICACAO.md, recontada: 1.428.041 pontos; 52.829 fora do contorno + 20 cm, 55.545 fora de h em [-10, 278], 36.416 flutuantes; limiar 2,284 cm = 1,452 + 2 x 0,416; 1.283.251 mantidos; confere com malha_limpa/resumo.json).
- sfm/points3d.npy + report_filtragem/outlier_mask_std2.0.npy: 727.716 pontos, 9.511 removidos (1,31%); limiar recalculado 0,192 (0,043 + 2 x 0,075) separa 9.503 (8 de diferenca; histograma colorido pela mascara).
Fatia: h 100 a 110 cm, 91.997 pontos: 85.809 mantidos, 3.660 fora do contorno, 2.528 flutuantes, 0 por altura; 6.188 removidos (6,7%).
Corte vertical: faixa de 10 cm em x = -10 cm da sala, 24.933 pontos.
