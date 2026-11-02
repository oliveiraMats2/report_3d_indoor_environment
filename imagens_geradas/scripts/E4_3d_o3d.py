import numpy as np, pycolmap, open3d as o3d, matplotlib
from matplotlib import cm
R="/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha/"
rec=pycolmap.Reconstruction(R+"exports/colmap/sparse/0")
P=np.load(R+"scale/points3d_cm.npy"); C=np.load(R+"scale/centers_cm.npy"); tl=np.load(R+"sfm/track_len.npy")
fl=np.load(R+"planes/floor.npy"); n=fl[:3]/np.linalg.norm(fl[:3]); d=fl[3]/np.linalg.norm(fl[:3])
if np.median(P@n+d)<0: n,d=-n,-d
h=P@n+d
ins=(h>0)&(h<285)
ref=np.median(C,0); a=np.cross(n,[1,0,0]);a/=np.linalg.norm(a);b=np.cross(n,a)
Q=P[ins]-ref;w,v=np.linalg.eigh(np.cov(np.c_[Q@a,Q@b].T));e1=v[0,1]*a+v[1,1]*b;e2=np.cross(n,e1)
M=np.c_[e1,e2,n]; tf=lambda X:(X-ref)@M
Pt=tf(P[ins]);t=tl[ins]
lo=np.percentile(Pt[:,:2],1.5,0);up=np.percentile(Pt[:,:2],98.5,0)
k=((Pt[:,:2]>lo)&(Pt[:,:2]<up)).all(1);Pt,t=Pt[k],t[k]
rng=np.random.default_rng(0);i=rng.choice(len(Pt),300000,replace=False);Pt,t=Pt[i],t[i]
col=cm.viridis((np.clip(t,3,30)-3)/27)[:,:3]
pc=o3d.geometry.PointCloud();pc.points=o3d.utility.Vector3dVector(Pt);pc.colors=o3d.utility.Vector3dVector(col)
# câmeras via pycolmap
ims=sorted(rec.images.values(),key=lambda im:im.name)
def pose(im):
    cw=im.cam_from_world(); cw=cw() if callable(cw) else cw
    Rm=cw.rotation.matrix(); tv=cw.translation
    return Rm,-Rm.T@tv
# escala model->cm
c0=pose(ims[0])[1]; names=list(np.load(R+"sfm/image_names.npy"))
cm0=C[names.index(ims[0].name)]; c1=pose(ims[500])[1]; cm1=C[names.index(ims[500].name)]
s=np.linalg.norm(cm1-cm0)/np.linalg.norm(c1-c0); print("escala",s)
SHOW="img_1271.png"; L=15.0
pts=[];lines=[];cols=[];
def frus(Rm,c,colr):
    cc=tf(c*s*0+ (c*s+ (C[0]-c0*s)))  # alinha origem
    return cc
off=C[0]-c0*s
for j,im in enumerate(ims):
    if j%40 and im.name!=SHOW: continue
    Rm,c=pose(im); cc=tf(c*s+off)
    ax=Rm[2]@M; r=Rm[0]@M; dn=Rm[1]@M
    z=L; hw=z*0.55*1.0; hh=hw*9/16
    cs=[cc+z*ax+sx*hw*r+sy*hh*dn for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    base=len(pts);pts+= [cc]+cs
    dd=Pt.mean(0)-cc; print('cos',float(ax@dd/np.linalg.norm(dd)))
    ln=[[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    lines+=[[base+x,base+y] for x,y in ln]
    cols+=[[1,.5,0] if im.name==SHOW else [0,.2,1]]*8
ls=o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.array(pts)),o3d.utility.Vector2iVector(lines));ls.colors=o3d.utility.Vector3dVector(cols)
W,H=2200,1900
r=o3d.visualization.rendering.OffscreenRenderer(W,H)
r.scene.set_background([1,1,1,1]);r.scene.view.set_post_processing(False)
m=o3d.visualization.rendering.MaterialRecord();m.shader="defaultUnlit";m.point_size=3
ml=o3d.visualization.rendering.MaterialRecord();ml.shader="unlitLine";ml.line_width=5
r.scene.add_geometry("pc",pc,m);r.scene.add_geometry("ls",ls,ml)
ctr=np.array([0,0,60.]);ext=np.array([up[0]-lo[0],up[1]-lo[1]]).max()
import sys
az=np.radians(float(sys.argv[1]) if len(sys.argv)>1 else -60);el=np.radians(38)
eye=ctr+ext*1.45*np.array([np.cos(az)*np.cos(el),np.sin(az)*np.cos(el),np.sin(el)])
r.scene.camera.set_projection(45,W/H,1,5000,o3d.visualization.rendering.Camera.FovType.Vertical)
r.scene.camera.look_at(ctr,eye,[0,0,1])
r.render_to_image()
img=r.render_to_image();o3d.io.write_image("/tmp/claude-1009/e4_3d.png",img)
