# E4_tie_points

Equivalente ao exemplo "image copy 4.png" (PIX4Dmatic, tie points e câmeras).

- Painel 3D: render Open3D (OffscreenRenderer, 2 renders, fundo branco, point_size 3) de 300000 pontos amostrados (semente 0) de scale/points3d_cm.npy. Cor = track_len (sfm/track_len.npy), viridis truncado em 30; colorbar desenhada em matplotlib. Teto removido (altura acima de 285 cm, chão de planes/floor.npy) e outliers fora dos percentis 1,5 a 98,5 em x,y. Script: scripts/E4_3d_o3d.py.
- Câmeras: frustums como LineSet (15 cm), 1 a cada 40 imagens, a partir de exports/colmap/sparse/0 via pycolmap (centro = -R^T t, escala 29,37 para cm); azul, e laranja para a câmera do quadro. Orientação conferida: 100% dos pontos 3D do quadro ficam à frente do eixo óptico.
- Quadro: img_1271.png (images.txt), 12013 de 15375 keypoints com point3D_id válido.
- Caixa: 2326/2327 imagens, 727716 tie points, track médio 9,41, erro 0,565 px (valores fornecidos na tarefa).
- Composição final: scripts/E4_tie_points.py (lê o render salvo em /tmp/claude-1009/e4_3d.png, gerado por E4_3d_o3d.py).
