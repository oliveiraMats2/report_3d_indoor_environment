"""G11: nuvem limpa da cozinha (sem teto), sem a parede da janela, vista obliqua ~60 graus.
Entrada: 3d_indoor_environment/cozinha/results/config_cozinha/malha_limpa/nuvem_limpa.ply, mesh/room_fill/T_room.npy"""
import sys, numpy as np, open3d as o3d
from open3d.visualization import rendering
from PIL import Image
C="/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha"
OUT="/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/G11_vista_cima_sem_teto.png"
XCUT=-138.0   # parede da janela: x_sala < XCUT (inclui a margem externa)
W,H,PS=2400,1800,float(sys.argv[1]) if len(sys.argv)>1 else 5.0
AZ=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
p=o3d.io.read_point_cloud(C+"/malha_limpa/nuvem_limpa.ply")
P=np.asarray(p.points);K=np.asarray(p.colors)
T=np.load(C+"/mesh/room_fill/T_room.npy")
Q=(np.linalg.inv(T)@np.c_[P,np.ones(len(P))].T).T[:,:3]
keep=Q[:,0]>=XCUT
Q,K=Q[keep],K[keep]
print("mantidos",keep.sum(),"removidos",(~keep).sum())
# eixos: x,y no piso, h para cima -> usar z=h
pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(Q);pc.colors=o3d.utility.Vector3dVector(K)
r=rendering.OffscreenRenderer(2*W,2*H)
m=rendering.MaterialRecord();m.shader="defaultUnlit";m.point_size=PS*2
r.scene.set_background([1,1,1,1]);r.scene.add_geometry("p",pc,m)
r.scene.view.set_post_processing(False)
c=np.array([0,-20,70.]);el=np.radians(60);d=560
a=np.radians(AZ)  # AZ=0: camera no lado -x olhando para +x
eye=c+d*np.array([-np.cos(el)*np.cos(a),-np.cos(el)*np.sin(a),np.sin(el)])
r.scene.camera.set_projection(50,W/H,10,3000,rendering.Camera.FovType.Vertical)
r.scene.camera.look_at(c,eye,[0,0,1])
r.render_to_image()
im=np.asarray(r.render_to_image())
Image.fromarray(im).resize((W,H),Image.LANCZOS).save(OUT,dpi=(250,250))
