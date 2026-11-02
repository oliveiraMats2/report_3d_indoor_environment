# G11 vista de cima sem teto
Fonte: 3d_indoor_environment/cozinha/results/config_cozinha/malha_limpa/nuvem_limpa.ply (1.283.251 pontos, sem teto h>278 cm e sem exterior, ver ESPECIFICACAO.md), T_room em mesh/room_fill/T_room.npy.
Removido adicionalmente: pontos com x_sala < -138 cm (parede da janela mais margem externa/vegetacao): 248.785 pontos; restam 1.034.466. A parede foi escolhida por concentrar os pontos verdes da vegetacao externa (x_sala < -100) e ficar rente ao lado da camera; nao ha rotulo "janela" em planes/, entao e inferencia.
Render Open3D Offscreen 4800x3600 (point_size 10), reducao LANCZOS para 2400x1800, elevacao 60 graus, fundo branco. Script: scripts/G11_vista_cima_sem_teto.py (args: point_size azimute).
