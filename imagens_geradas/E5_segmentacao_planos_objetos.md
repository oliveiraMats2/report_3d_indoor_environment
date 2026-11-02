# E5 segmentacao planos + objetos (figura extra, fora do relatorio)

Segmentacao GEOMETRICA (planos + agrupamento), nao deteccao semantica. Nao ha rotulos de classe (mesa, cadeira etc.).

Dados: cozinha/results/config_cozinha/malha_limpa/nuvem_limpa.ply (1.283.251 pts, voxel 2 cm), planes/floor.npy, walls.npy (5 planos), T_floor.npy.
Planos: piso + 5 paredes (Parede 3 e o chanfro; angulos medidos em artifacts/config_cozinha/wall_angles.csv: 141,5 e 132,6 graus contra 135 esperado). Pontos a menos de 4 cm de cada plano.
Clusters: pontos com 4 cm < h < 278 cm, a mais de 10 cm de qualquer plano; DBSCAN (Open3D) eps 3 cm, min 8 pts; mantidos >= 250 pts e maior lado >= 25 cm: 11 clusters (de 107).
Clusters com lado > 200 cm foram separados em h < 135 e h >= 135 cm e reagrupados (lacuna de pontos entre 130 e 205 cm no perfil). C1 (423 x 255 x 131 cm) e o conjunto conectado dos armarios baixos em L, nao um objeto unico.
Caixa: orientada em torno do eixo vertical (rotacao no plano do piso de area minima), nao OBB 3D completa.
Script: scripts/E5_segmentacao_planos_objetos.py (args: azimute elevacao, padrao 200 55). Teto nao exibido.
