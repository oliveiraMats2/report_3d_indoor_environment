# G15 wireframe da malha

Arquivo: G15_wireframe_malha.png (3520x1804 px, 220 dpi). Script: scripts/G15_wireframe_malha.py.

Fonte: 3d_indoor_environment/cozinha/results/config_cozinha/malha_limpa/malha_limpa.ply (3.444.576 vértices, 6.888.396 faces, malha_limpa/resumo.json), levada ao referencial da sala com inv(mesh/room_fill/T_room.npy).

- Painel (a): simplificação por quadric decimation (Open3D, alvo 40.000 faces; resultado 51.516 faces e 15.439 vértices, malha final desenhada). Arestas em linhas finas escuras sobre fundo branco, vista oblíqua de cima, sem teto (a malha limpa já é cortada em h = 278 cm). Furos = regiões sem superfície reconstruída.
- Painel (b): zoom na malha densa original (sem simplificação), caixa de 45 cm de lado, x de -170 a -125, y de 62 a 107, h de 70 a 115 cm (referencial da sala), 96.982 faces; a caixa está marcada em vermelho em (a).
- Linhas ocultas removidas por malha branca opaca; linhas deslocadas levemente para a câmera.
