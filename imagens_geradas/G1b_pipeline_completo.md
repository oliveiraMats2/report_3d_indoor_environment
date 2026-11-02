# G1b Pipeline completo: SfM, densa e malha, comparação com o LiDAR (figura extra)

O que é: cópia estendida do G1 (G1_pipeline_sfm.png, que não foi alterado). 15 pol, 250 dpi, 3750x3507 px. Mesmo estilo: painéis com moldura, setas pretas, título e uma linha de número por painel. A faixa do título tem uma cor clara por trilha (legenda no topo): azul = trilha esparsa (SfM), laranja = trilha densa e malha, verde = avaliação com o LiDAR.

Leitura em serpente: fileira 1 da esquerda para a direita (quadros, SIFT, correspondências, poses); desce; fileira 2 da direita para a esquerda (triangulação, ajuste de feixes, mapas de profundidade, fusão); desce; fileira 3 da esquerda para a direita (limpeza, Poisson, detalhe da malha, reconstrução final); desce para a faixa larga do LiDAR.

R = 3d_indoor_environment/cozinha/; RUN = R/results/config_cozinha/.

## Painéis herdados do G1 (mesmas fontes, ver G1_pipeline_sfm.md)
- Quadros de entrada: agora 9 quadros (img_0060, 0720, 0800, 1110, 1140, 1320, 1560, 1950, 2040) em 3 x 3; 2.327 quadros (artifacts/config_cozinha/extraction_metrics.csv).
- Extração SIFT (score 0,936), Correspondências (666 de 765 inliers), Estimativa das poses (2.326 poses), Triangulação (ponto 3046041), Ajuste de feixes (erro médio 0,565 px): iguais ao G1, só redimensionados.
- Reconstrução final: agora renderizada de novo (2000x1640, fundo branco) a partir de RUN/malha_limpa/malha_limpa.ply com a mesma câmera de render_malha_limpa.py (AZ 0, EL 55, DIST 640, fov 55, centro = média dos vértices). 6,89 M faces = 6.888.396 (RUN/malha_limpa/resumo.json, "final"). Sem teto (corte em 278 cm) e sem exterior (contorno + 20 cm), RUN/malha_limpa/ESPECIFICACAO.md.

## Painéis novos
- Mapas de profundidade: quadro img_1110 (o mesmo do painel SIFT), imagem não distorcida (RUN/dense/undistorted/images/img_1110.png, 1600x896) à esquerda e mapa geométrico do PatchMatch à direita (RUN/dense/patch_match/ws/stereo/depth_maps/img_1110.png.geometric.bin, formato COLMAP lido por images/report/scripts/profundidade.py::read_map, x 29,3745 cm/unidade). Escala turbo 50 a 250 cm, a mesma da faixa do LiDAR; preto = sem profundidade. 2.325 = número de arquivos *.geometric.bin na pasta.
- Fusão das profundidades: nuvem densa bruta RUN/dense/patch_match/points.npy (98.083.833 pontos, lida por mmap; o fused.ply de 5 GB não foi aberto), subamostra 1 a cada 30 (3,27 M pontos) com colors.npy, x 29,3745 cm, mesma câmera do painel final, corte de visualização acima de 255 cm do piso (planes/floor.npy) para ver dentro.
- Limpeza da nuvem: RUN/clean/patch_match/points.npy (1.428.041 pontos, voxel 1 cm); mantidos = pontos presentes em RUN/malha_limpa/nuvem_limpa.ply (1.283.251, comparação exata de coordenadas), espúrios = o resto; o script confere 144.790 = resumo.json ("nuvem"/"espurios"), 10,1% de 1.428.041. Verde modulado pelo brilho da cor real; vermelho = espúrio. Mesma câmera e corte de 255 cm (os espúrios acima do teto, parte dos 55.545 fora da faixa de altura, não aparecem na vista).
- Malha de Poisson: RUN/report_filtragem/poisson_q0.00.ply (1.555.804 vértices = 1,56 M, report_filtragem/filtragem.csv, linha poisson), em vermelho os vértices que o corte por densidade q = 0,02 remove (ausentes em poisson_q0.02.ply), como images/report/malha_antes.jpg; corte de visualização em 255 cm. Atenção: essa malha é a do relatório (fig:limpeza), gerada da nuvem voxelizada sem a remoção de espúrios; a malha final (3.592.082 vértices no Poisson) vem da nuvem limpa.
- Detalhe da malha: malha limpa sem simplificação (malha_limpa.ply) no referencial da sala (inv(RUN/mesh/room_fill/T_room.npy)), caixa de 20 cm centrada em (x, y, h) = (-132,9; -130,1; 108,9) cm, potes azuis da bancada (vistos em img_1620); 17.815 triângulos inteiros na caixa. Projeção ortográfica olhando do meio da sala para a bancada, 34 graus de cima; cada face pintada com a média das cores dos vértices x sombreamento lambertiano, arestas pretas finas; barra de 2 cm.
- Comparação com o LiDAR (faixa larga):
  - nuvem densa x LiDAR (planta): RUN/clean/patch_match/points.npy levada ao LiDAR por 3d_indoor_environment_report/outputs/t5_lidar/run/lidar/T_clean_patch_match.npy (similaridade, escala 1,0273), distância de cada ponto à malha 3d_indoor_environment/cozinha_lidar/fused/mesh_closed_2M.ply (Open3D RaycastingScene.compute_distance, dividida pela escala), vista de cima sem o teto (y do LiDAR < 120 cm), escala magma 0 a 20 cm.
  - quadro RGB, profundidade reconstruída, profundidade LiDAR e |reconstrução − LiDAR| no quadro img_1740: mesmas funções e escalas de images/report/scripts/profundidade.py (G7, profundidade_rgb_recon_lidar.jpg), agora na resolução cheia 1600x896.
  - Números da legenda: notas/t5_lidar.md, coluna patch_match: Chamfer 3,82 cm; F-score a 5 cm 0,792 (0,79); 93,5% dos pontos do modelo a até 5 cm do LiDAR.

Script: scripts/G1b_pipeline_completo.py (cópia estendida de scripts/G1_pipeline_sfm.py; G1B_CACHE=<pasta> guarda os renders 3D entre execuções).
