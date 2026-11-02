import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
R="/home/moliveira/Documents/3d_indoor_environment/cozinha/"
S=R+"results/config_cozinha/"
OUT="/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/E4_tie_points.png"
P=np.load(S+"scale/points3d_cm.npy"); C=np.load(S+"scale/centers_cm.npy")
tl=np.load(S+"sfm/track_len.npy"); Rw=np.load(S+"sfm/cam_R.npy"); sc=np.load(S+"scale/scale.npy")[0]
fl=np.load(S+"planes/floor.npy"); n=fl[:3]/np.linalg.norm(fl[:3]); d=fl[3]/np.linalg.norm(fl[:3])
h=P@n+d  # altura acima do chão (cm)
if np.median(h)<0: n,d,h=-n,-d,-h
# base horizontal por PCA dos pontos internos
ins=(h>0)&(h<285)&(np.abs(P-np.median(P,0))<400).all(1)
Hc=C@n+d
ref=np.median(C,0)
a=np.cross(n,[1,0,0]); a/=np.linalg.norm(a); b=np.cross(n,a)
Q=P[ins]-ref; q2=np.c_[Q@a,Q@b]; w,v=np.linalg.eigh(np.cov(q2.T)); 
e1=v[0,1]*a+v[1,1]*b; e2=np.cross(n,e1)
M=np.c_[e1,e2,n]
def tf(X): return (X-ref)@M
Pi=tf(P[ins]); ti=tl[ins]; hi=h[ins]
# remove teto: aqui h<285 já remove; mantém chão e paredes
rng=np.random.default_rng(0); idx=rng.choice(len(Pi),160000,replace=False)
Pi,ti,hi=Pi[idx],ti[idx],hi[idx]
lo=np.percentile(Pi[:,:2],1.5,axis=0);up=np.percentile(Pi[:,:2],98.5,axis=0)
k=((Pi[:,:2]>lo)&(Pi[:,:2]<up)).all(1);Pi,ti,hi=Pi[k],ti[k],hi[k]
Cc=tf(C)
# quadro
names=np.load(S+"sfm/image_names.npy")
txt=open(S+"exports/colmap/sparse/0/images.txt").read().splitlines()[4:]
best=None;cnt={}
for i in range(0,len(txt),2):
    nm=txt[i].split()[-1]; f=np.array(txt[i+1].split(),float).reshape(-1,3)
    cnt[nm]=(int((f[:,2]>=0).sum()),len(f),f)
mid=len(names)//2
nm=sorted(cnt,key=lambda k:-cnt[k][0] if abs(int(k[4:8])-1300)<40 else 0)[0]
nt,nk,f=cnt[nm]; ok=f[:,2]>=0
img=np.array(Image.open(R+"frames/all_stride10/"+nm).convert("RGB"))
ic=int(nm[4:8]); 
# figura
fig=plt.figure(figsize=(16,8.2),dpi=220)
from PIL import Image as _I
im3=np.array(_I.open("/tmp/claude-1009/e4_3d.png").convert("RGB"))
m_=(im3<245).any(2);ys,xs=np.where(m_);im3=im3[ys.min()-10:ys.max()+10,xs.min()-10:xs.max()+10]
ax=fig.add_axes([0.0,0.02,0.58,0.9]);ax.imshow(im3);ax.axis("off")
ax.set_title("Nuvem esparsa de tie points e posições das câmeras (teto removido)",fontsize=12)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color="#0033ff",lw=2,label="câmeras (1 a cada 40)"),Line2D([0],[0],color="#ff8000",lw=2,label=f"câmera do quadro ({nm})")],loc="upper left",fontsize=9)
import matplotlib.cm as _cm,matplotlib.colors as _c
cax=fig.add_axes([0.6,0.2,0.01,0.5]);cb=fig.colorbar(_cm.ScalarMappable(_c.Normalize(3,30),"viridis"),cax=cax)
cb.set_label("observações por tie point (truncado em 30)")
# painel direito
a2=fig.add_axes([0.65,0.50,0.34,0.42]); a2.imshow(img); 
a2.scatter(f[~ok,0],f[~ok,1],s=3,c="#888",alpha=0.5,label="keypoints sem ponto 3D")
a2.scatter(f[ok,0],f[ok,1],s=2,c="#2ca02c",label=f"viraram tie points ({nt})")
a2.set_xlim(0,img.shape[1]);a2.set_ylim(img.shape[0],0);a2.axis("off")
a2.set_title(f"{nm}: {nt} de {nk} keypoints com ponto 3D válido",fontsize=11)
a2.legend(loc="lower right",fontsize=8,framealpha=0.9)
a3=fig.add_axes([0.65,0.06,0.34,0.34]); a3.axis("off")
a3.add_patch(Rectangle((0,0),1,1,fc="#f4f4f4",ec="#333",transform=a3.transAxes))
st=["Estatísticas da reconstrução (SfM)","","Imagens registradas: 2326/2327","Tie points: 727716","Track médio: 9,41 observações","Erro de reprojeção médio: 0,565 px"]
for k,s in enumerate(st): a3.text(0.06,0.88-k*0.14,s,fontsize=13 if k else 14,fontweight="bold" if k==0 else "normal",transform=a3.transAxes,va="top")
fig.savefig(OUT,dpi=220); print(nm,nt,nk,len(Pi),ti.mean(),Hc.min(),Hc.max())
