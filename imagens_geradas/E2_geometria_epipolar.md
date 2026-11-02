# E2 geometria epipolar (esquema didático, figura extra, fora de escala)
Script: scripts/E2_geometria_epipolar.py (rodar com o python do .venv do projeto; variáveis opcionais NA, NB, TILT, T1, T2).
Dados: modelo COLMAP cozinha/results/config_cozinha/exports/colmap/sparse/0 (29,3745 cm/unidade); quadros cozinha/frames/all_stride10/img_0771.png e img_0789.png (sem distorção).
Par escolhido: img_0771 e img_0789. Busca em pares próximos do modelo (distância de índice < 80, 20 a 80 cm de base, mais de 350 pontos 3D em comum) com epipolos de lado oposto e deslocamento lateral: este tem 2092 pontos 3D em comum, linha de base 51,9 cm e linhas epipolares quase horizontais. O par 0774/0790 (tinha 2106 em comum) tem movimento mais frontal e epipolos mais distantes; nenhum par tinha epipolos dentro do quadro e laterais ao mesmo tempo.
Ponto 3D: id 3146577, trilha 91, erro de reprojeção 0,91 px; profundidade 129,9 cm (cam. 1) e 135,5 cm (cam. 2).
x1 = (704,0; 466,7) px, x2 = (993,8; 433,1) px (sem distorção); distância de x2 à linha epipolar F·x1: 0,77 px.
Epipolos (núcleo de F): e1 = (6273; 817) px, e2 = (-4134; 632) px, fora do quadro 1920x1080.
Miniaturas: recorte 560x340 px em torno de x1 e x2. Posições e inclinações dos planos, de C1, C2 e X no desenho são livres; epipolos marcados onde a linha epipolar desenhada cruza a linha de base.
