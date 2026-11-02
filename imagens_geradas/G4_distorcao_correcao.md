# G4_distorcao_correcao

Figura extra (nao entra no relatorio). Passo a passo da correcao de distorcao no quadro cozinha/frames/all_stride10/img_1320.png (janela com grade e azulejos).
Intrinsecos e distorcao lidos de cozinha/results/config_cozinha/exports/colmap/sparse/0/cameras.bin (OPENCV 1920x1080).
Magnitudes (px, modelo OpenCV aplicado a cada pixel ideal): centro 0,00; meio da borda esquerda 14,0; meio da borda superior 7,5; canto sup.-esq. 5,7; maximo 16,2 em (1659,0).
Paineis (4) e (5): o zoom e real; o painel (5) multiplica k1,k2,p1,p2 por 8, so para visualizacao.
Rodar: /home/moliveira/Documents/3d_indoor_environment/.venv/bin/python scripts/G4_distorcao_correcao.py
