"""E5: segmentacao geometrica (planos RANSAC + DBSCAN nos pontos restantes). Somente leitura nos dados."""
import sys, json, numpy as np, open3d as o3d, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha/'
OUT='/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/E5_segmentacao_planos_objetos.png'
AZ=float(sys.argv[1]) if len(sys.argv)>1 else 200; EL=float(sys.argv[2]) if len(sys.argv)>2 else 55
VOX=2.0; TOL_PLANO=4.0; MARGEM=10.0; HMAX=278.0; EPS=3.0; MINPTS=8; MIN_VOX=250; MIN_DIM=25.0
pc=o3d.io.read_point_cloud(R+'malha_limpa/nuvem_limpa.ply').voxel_down_sample(VOX)
P=np.asarray(pc.points); T=np.load(R+'planes/T_floor.npy'); Q=P@T[:3,:3].T+T[:3,3]   # x,y no piso; z = altura (cm)
fl=np.load(R+'planes/floor.npy'); W=np.load(R+'planes/walls.npy')
h=Q[:,2]; D=np.stack([P@w[:3]+w[3] for w in W],1); inside=(D>-TOL_PLANO).all(1)
lab=np.full(len(P),-1)   # -1 resto, 0 piso, 1..5 paredes
lab[inside&(abs(h)<TOL_PLANO)]=0
wallm=inside&(h>=TOL_PLANO)&(h<=HMAX)&(abs(D).min(1)<TOL_PLANO)
lab[wallm]=1+np.argmin(abs(D),1)[wallm]
rest=inside&(lab==-1)&(h>4)&(h<HMAX)&(D.min(1)>MARGEM)
q=o3d.geometry.PointCloud(); q.points=o3d.utility.Vector3dVector(Q[rest])
cl=np.array(q.cluster_dbscan(EPS,MINPTS)); Qr=Q[rest]
def yawbox(X):
    best=None
    for a in np.deg2rad(np.arange(0,90,1)):
        c,s=np.cos(a),np.sin(a); u=X[:,0]*c+X[:,1]*s; v=-X[:,0]*s+X[:,1]*c
        A=np.ptp(u)*np.ptp(v)
        if best is None or A<best[0]: best=(A,a,u.min(),u.max(),v.min(),v.max())
    _,a,u0,u1,v0,v1=best; c,s=np.cos(a),np.sin(a)
    z0,z1=X[:,2].min(),X[:,2].max()
    cor=[(c*u-s*v,s*u+c*v) for u,v in [(u0,v0),(u1,v0),(u1,v1),(u0,v1)]]
    return np.array([[x,y,z] for z in (z0,z1) for x,y in cor]),(u1-u0,v1-v0,z1-z0)
# clusters gigantes (maior dimensao > 200 cm): separa em faixas de altura (h < 135 e h >= 135 cm; lacuna 130-205 cm vista no perfil) e reagrupa
GRANDE=200.0; HCORTE=135.0; grupos=[]
for k in range(cl.max()+1):
    X=Qr[cl==k]
    if len(X)>=MIN_VOX and np.ptp(X,0).max()>GRANDE:
        for sel in (X[:,2]<HCORTE, X[:,2]>=HCORTE):
            Y=X[sel]
            if len(Y)==0: continue
            qq=o3d.geometry.PointCloud(); qq.points=o3d.utility.Vector3dVector(Y); l2=np.array(qq.cluster_dbscan(EPS,MINPTS))
            for j in range(l2.max()+1): grupos.append(Y[l2==j])
    else: grupos.append(X)
objs=[]
for X in grupos:
    if len(X)<MIN_VOX: continue
    B,dim=yawbox(X)
    if max(dim[0],dim[1])<MIN_DIM: continue
    objs.append(dict(n=len(X),box=B,dim=dim,pts=X))
objs.sort(key=lambda o:-o['n'])
print('clusters totais',cl.max()+1,'mantidos',len(objs),[ (o['n'],[round(d) for d in o['dim']]) for o in objs])
# cores
cpl={0:(0.80,0.80,0.80),1:(0.55,0.75,0.95),2:(0.98,0.82,0.55),3:(0.70,0.88,0.70),4:(0.95,0.70,0.80),5:(0.80,0.72,0.92)}
cob=[(0.85,0.10,0.10),(0.10,0.45,0.85),(0.95,0.55,0.0),(0.1,0.65,0.35),(0.6,0.1,0.7),(0.0,0.7,0.7),(0.7,0.5,0.0),(0.9,0.2,0.6)]
G=2.2  # o renderizador aplica gama; pre-compensa
cols=np.zeros((len(Q),3))
for k,c in cpl.items(): cols[lab==k]=np.array(c)**G
geoms=[]
pw=o3d.geometry.PointCloud(); m=lab>=0; pw.points=o3d.utility.Vector3dVector(Q[m]); pw.colors=o3d.utility.Vector3dVector(cols[m])
pcs=[pw]
for i,o in enumerate(objs):
    g=o3d.geometry.PointCloud(); g.points=o3d.utility.Vector3dVector(o['pts']); g.colors=o3d.utility.Vector3dVector(np.tile(np.array(cob[i%len(cob)])**G,(len(o['pts']),1))); pcs.append(g)
ls=o3d.geometry.LineSet(); E=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
pts=[];lines=[];lc=[]
for i,o in enumerate(objs):
    b=len(pts); pts+=list(o['box']); lines+=[[b+a,b+c] for a,c in E]; lc+=[np.array(cob[i%len(cob)])**G]*12
ls.points=o3d.utility.Vector3dVector(np.array(pts)); ls.lines=o3d.utility.Vector2iVector(lines); ls.colors=o3d.utility.Vector3dVector(lc)
Wd,Ht=2000,1500
r=o3d.visualization.rendering.OffscreenRenderer(Wd,Ht); r.scene.set_background([1,1,1,1]); r.scene.show_skybox(False); r.scene.set_lighting(r.scene.LightingProfile.NO_SHADOWS,(0,0,-1))
mp=o3d.visualization.rendering.MaterialRecord(); mp.shader='defaultUnlit'; mp.point_size=6.0
ml=o3d.visualization.rendering.MaterialRecord(); ml.shader='unlitLine'; ml.line_width=4.0
for i,g in enumerate(pcs): r.scene.add_geometry(f'p{i}',g,mp)
r.scene.add_geometry('box',ls,ml)
ctr=np.array([Qr[:,0].mean()*0+25,0,110.]); ctr[:2]=(Q[lab>=0][:,:2].min(0)+Q[lab>=0][:,:2].max(0))/2
az,el=np.deg2rad(AZ),np.deg2rad(EL); dist=720
eye=ctr+dist*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
f=1500.; K=o3d.camera.PinholeCameraIntrinsic(Wd,Ht,f,f,Wd/2,Ht/2)
z=(ctr-eye)/np.linalg.norm(ctr-eye); x=np.cross(z,[0,0,1]); x/=np.linalg.norm(x); y=np.cross(z,x)
Rm=np.stack([x,y,z]); Ex=np.eye(4); Ex[:3,:3]=Rm; Ex[:3,3]=-Rm@eye
r.setup_camera(K,Ex)
r.render_to_image(); img=np.asarray(r.render_to_image())
def proj(p):
    c=Rm@(np.asarray(p)-eye); return np.array([f*c[0]/c[2]+Wd/2,f*c[1]/c[2]+Ht/2])
bg=img[5,5].copy(); img=img.copy(); img[(abs(img.astype(int)-bg.astype(int)).sum(2)<6)]=255; fig=plt.figure(figsize=(13,9.2),dpi=220); ax=fig.add_axes([0,0.0,0.74,0.93]); ax.imshow(img); ax.axis('off')
fig.text(0.01,0.965,'Segmentação geométrica da cozinha: planos RANSAC (piso e 5 paredes) + agrupamento DBSCAN dos pontos restantes',fontsize=12.5,weight='bold',va='center')
fig.text(0.01,0.935,'Não é detecção semântica: os clusters são grupos de pontos acima do piso e fora dos planos, com caixa delimitadora orientada (OBB, eixo vertical) e dimensões em cm',fontsize=9.5,va='center',color='0.25')
# rotulos de planos (centroide)
nomes={1:'Parede 1',2:'Parede 2',3:'Parede 3 (chanfro)',4:'Parede 4',5:'Parede 5',0:'Piso'}
for k,t in nomes.items():
    X=Q[lab==k]
    if k>0: X=X[(X[:,2]>140)&(X[:,2]<200)] if ((X[:,2]>140)&(X[:,2]<200)).sum()>50 else X
    c=np.median(X,0); u=proj(c)
    ax.text(u[0],u[1],t,fontsize=9,ha='center',va='center',weight='bold',bbox=dict(fc='white',ec='0.4',lw=0.6,alpha=0.85,boxstyle='round,pad=0.2'))
for i,o in enumerate(objs):
    top=o['box'][4:].mean(0); u=proj(top); c=cob[i%len(cob)]
    ax.text(u[0],u[1]-25,f'C{i+1}',fontsize=10,weight='bold',color='white',ha='center',va='center',bbox=dict(fc=c,ec='none',boxstyle='round,pad=0.2'))
# legenda lateral
ly=0.86; ax2=fig.add_axes([0.745,0.0,0.255,0.93]); ax2.axis('off'); ax2.set_xlim(0,1); ax2.set_ylim(0,1)
y0=0.97; ax2.text(0,y0,'Planos (RANSAC, tolerância 4 cm)',fontsize=10,weight='bold',va='top'); y0-=0.04
for k in range(6):
    n=int((lab==k).sum()); ax2.add_patch(plt.Rectangle((0,y0-0.012),0.06,0.022,fc=cpl[k],ec='0.4',lw=0.5))
    ax2.text(0.09,y0,f'{nomes[k]}: {n} pts',fontsize=8.5,va='center'); y0-=0.032
y0-=0.02; ax2.text(0,y0,f'Clusters (DBSCAN, {len(objs)} mantidos)',fontsize=10,weight='bold',va='top'); y0-=0.04
ax2.text(0,y0+0.005,'dimensões da caixa: L x P x A (cm)',fontsize=8,color='0.3',va='center'); y0-=0.03
for i,o in enumerate(objs):
    d=o['dim']; ax2.add_patch(plt.Rectangle((0,y0-0.012),0.06,0.022,fc=cob[i%len(cob)],ec='none'))
    ax2.text(0.09,y0,f'C{i+1}: {max(d[:2]):.0f} x {min(d[:2]):.0f} x {d[2]:.0f}  ({o["n"]} pts)',fontsize=8.5,va='center'); y0-=0.032
y0-=0.02
par=f'Parâmetros: voxel {VOX:g} cm; planos de planes/*.npy, |d| < {TOL_PLANO:g} cm; resto = h > 4 cm,\nh < {HMAX:g} cm, a mais de {MARGEM:g} cm de qualquer plano;\nDBSCAN eps {EPS:g} cm, min {MINPTS} pts; filtro: >= {MIN_VOX} pts e\nmaior lado >= {MIN_DIM:g} cm. Cluster com lado > {GRANDE:g} cm\né separado em h < {HCORTE:g} e h >= {HCORTE:g} cm e reagrupado.\nTeto não exibido.'
ax2.text(0,y0,par,fontsize=7.8,va='top',color='0.25')
fig.text(0.01,0.008,'Fonte: cozinha/results/config_cozinha/malha_limpa/nuvem_limpa.ply, planes/{floor,walls,T_floor}.npy',fontsize=7.5,color='0.35')
fig.savefig(OUT,dpi=220); print(OUT)
