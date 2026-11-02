# G1 Pipeline SfM em 7 painéis (figura extra)

O que é: versão grande (12 pol, 300 dpi, 3600x2040 px) do diagrama de exemplo, só com dados da cozinha. Leitura: fileira de cima da esquerda para a direita (quadros, SIFT, correspondências, poses), seta desce, fileira de baixo da direita para a esquerda (triangulação, ajuste de feixes, reconstrução final).

Painéis e fontes (R = 3d_indoor_environment/cozinha/):
- Quadros: 5 de 2.327 (artifacts/config_cozinha/extraction_metrics.csv; frames/all_stride10/).
- SIFT: img_1110, 5000 keypoints redetectados, conferidos com detection_metrics.csv; score 0,936 (detector_choice.csv).
- Correspondências: par img_0774/img_0790, 666 inliers de 765 (matching_metrics.csv; matching/sift e epipolar/sift). Verde inliers, vermelho outliers.
- Poses: 2.326 centros (results/config_cozinha/scale/centers_cm.npy) e frustums em planta (sfm/cam_R.npy), sobre a nuvem esparsa limpa.
- Triangulação: ponto 3046041 (quina do livro), img_1150 e img_1154 (exports/colmap/sparse/0).
- Ajuste de feixes: 727.716 pontos (sfm/points3d.npy x 29,3745 cm/unidade) coloridos por sfm/point_error.npy (média 0,565 px; escala de cor até o percentil 99, 1,8 px).
- Reconstrução final: malha_limpa/render_malha_limpa.jpg (malha limpa, 6.888.396 faces, malha_limpa/resumo.json).

Script: scripts/G1_pipeline_sfm.py (adaptado de images/report/scripts/pipeline.py).
