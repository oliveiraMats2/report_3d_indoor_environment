# G3_lidar_malha_cinza

Figura extra (nao entra no relatorio). Malha LiDAR da cozinha em cinza, sem textura, vista obliqua de cima sem o teto, no estilo de `diagramas_exemplo/image copy 7.png`.

- Fonte: `3d_indoor_environment/cozinha_lidar/fused/mesh_closed_2M.ply` (nao usa `cozinha_merged.ply`).
- Malha original: 953.412 vertices e 1.999.694 faces (metros, y para cima).
- Recorte: caixa x em [-5,9; -1,9], z em [-3,0; 0,4] e y em [-1,05; 1,0] m (corte por altura remove teto e topo das paredes; y<-1,05 remove excesso abaixo do piso). Mantem triangulos com os 3 vertices dentro. Resultado: 603.827 vertices e 1.265.590 faces (saida do script).
- Cor descartada; normais por vertice recalculadas; material cinza (0,78), luz de sol direcional, fundo branco (mascara por profundidade). Camera a ~5,5 m de altura, ligeiramente inclinada.
- Script: `scripts/G3_lidar_malha_cinza.py` (usa `.venv` do repositorio; argumento opcional = altura de corte em y). O recorte da cozinha e uma estimativa visual a partir da densidade de vertices; as bordas externas do recorte ainda trazem ruido da cena LiDAR.
- Saida: PNG renderizado a 3000x2000 e cortado nas margens (cerca de 1800x1400 px), gravado com 300 dpi nos metadados.
