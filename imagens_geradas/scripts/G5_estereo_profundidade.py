# G5: estereo retificado (img_1150/img_1154) vs PatchMatch. Somente leitura dos dados.
import numpy as np, cv2, pycolmap, struct, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R="/home/moliveira/Documents/3d_indoor_environment"
M=R+"/cozinha/results/config_cozinha"
OUT="/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/G5_estereo_profundidade.png"
S=29.3745  # cm por unidade (modelo)
A,B_="img_1150.png","img_1154.png"
rec=pycolmap.Reconstruction(M+"/dense/undistorted/sparse")
im={i.name:i for i in rec.images.values()}
cam=rec.cameras[1]; fx,fy,cx,cy=cam.params; W,H=cam.width,cam.height
K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
p1=im[A].cam_from_world(); p2=im[B_].cam_from_world()
R1,t1=p1.rotation.matrix(),p1.translation; R2,t2=p2.rotation.matrix(),p2.translation
Rr=R2@R1.T; tr=t2-Rr@t1
Bcm=np.linalg.norm(tr)*S
L=cv2.imread(M+"/dense/undistorted/images/"+A); Rt=cv2.imread(M+"/dense/undistorted/images/"+B_)
R1r,R2r,P1,P2,Q,_,_=cv2.stereoRectify(K,np.zeros(5),K,np.zeros(5),(W,H),Rr,tr.reshape(3,1),flags=0,alpha=-1)
# nova K comum, centrada no raio central original (f mantido)
r0=R1r@np.array([0,0,1.]); fN=(fx+fy)/2
P1=np.array([[fN,0,W/2-fN*r0[0]/r0[2],0],[0,fN,H/2-fN*r0[1]/r0[2],0],[0,0,1,0]])
P2=P1.copy(); P2[0,3]=-fN*np.linalg.norm(tr)
m1=cv2.initUndistortRectifyMap(K,np.zeros(5),R1r,P1,(W,H),cv2.CV_32FC1)
m2=cv2.initUndistortRectifyMap(K,np.zeros(5),R2r,P2,(W,H),cv2.CV_32FC1)
Lr=cv2.remap(L,*m1,cv2.INTER_LINEAR); Rr_img=cv2.remap(Rt,*m2,cv2.INTER_LINEAR)
f=P1[0,0]
ND=224; MIND=96  # faixa de busca 96-320 px (Z de 49 a 165 cm)
sg=cv2.StereoSGBM_create(minDisparity=MIND,numDisparities=ND,blockSize=5,P1=8*3*25,P2=64*3*25,
    disp12MaxDiff=2,uniquenessRatio=5,speckleWindowSize=200,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
cl=cv2.createCLAHE(2.0,(8,8))
g=lambda x:cl.apply(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY))
disp=sg.compute(g(Lr),g(Rr_img)).astype(np.float32)/16
disp=cv2.medianBlur(disp,5)
valid=(disp>MIND+1)&(disp<MIND+ND-2)&(cv2.cvtColor(Lr,cv2.COLOR_BGR2GRAY)>0)
Z=np.where(valid,f*(Bcm)/np.maximum(disp,1e-3),np.nan)  # cm, Z=f*B/d
# PatchMatch
def readbin(p):
    with open(p,"rb") as fh:
        w,h,c=[int(x) for x in fh.readline().split(b"&")[:3]]
        a=np.fromfile(fh,np.float32)
    return a.reshape((c,h,w),order="F").squeeze() if False else a.reshape(h,w) if c==1 else None
with open(M+"/dense/patch_match/ws/stereo/depth_maps/%s.geometric.bin"%A,"rb") as fh:
    w=h=c=0; hdr=b""
    while hdr.count(b"&")<3: hdr+=fh.read(1)
    w,h,c=[int(x) for x in hdr.split(b"&")[:3]]
    pm=np.fromfile(fh,np.float32).reshape((h,w))
pm=pm*S
# rectificada -> original: ponto 3D no quadro retificado -> quadro original da camera esquerda
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
Zr=Z.copy()
X=(xx-P1[0,2])*Zr/f; Y=(yy-P1[1,2])*Zr/f
Pc=R1r.T@np.stack([X.ravel(),Y.ravel(),Zr.ravel()])   # quadro original
zc=Pc[2].reshape(H,W); u=fx*Pc[0].reshape(H,W)/zc+cx; v=fy*Pc[1].reshape(H,W)/zc+cy
ui=np.round(u).astype(int); vi=np.round(v).astype(int)
ok=valid&np.isfinite(zc)&(ui>=0)&(ui<W)&(vi>=0)&(vi<H)
pm_at=np.full((H,W),np.nan,np.float32)
pm_at[ok]=pm[vi[ok],ui[ok]]
okc=ok&(pm_at>0)
d=(zc-pm_at); d[~okc]=np.nan
med=np.nanmedian(d); mad=np.nanmedian(np.abs(d-med)); medabs=np.nanmedian(np.abs(d))
rel=np.nanmedian(np.abs(d)/pm_at)*100
n=int(okc.sum()); print("f",f,"B",Bcm,"med",med,"medabs",medabs,"rel%",rel,"n",n,"valid%",valid.mean()*100)
print("Z pct",np.nanpercentile(Z,[5,50,95]),"pm pct",np.nanpercentile(pm[pm>0],[5,50,95]), "disp pct",np.percentile(disp[valid],[5,50,95]))
# mapa estereo em cm comparavel (z no quadro original) para a figura 5 e 6
zshow=np.where(okc,zc,np.nan)
# ---------- figura
from matplotlib.ticker import FuncFormatter
vc=lambda x:("%s"%x).replace(".",",")
def F(fmt,*a,m=False): return (fmt%a).replace(".","{,}" if m else ",")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
fig=plt.figure(figsize=(18,12),dpi=200)
gs=fig.add_gridspec(3,3,height_ratios=[1,1,1.05],hspace=0.30,wspace=0.34,left=0.03,right=0.985,top=0.955,bottom=0.03)
def title(ax,n,t):
    ax.set_title("%d  %s"%(n,t),loc="left",fontsize=13,fontweight="bold",color="#1a3c8f")
# 1 par retificado
ax=fig.add_subplot(gs[0,0:2]); title(ax,1,"Par retificado (img_1150 | img_1154), linhas epipolares horizontais")
pair=np.hstack([cv2.cvtColor(Lr,cv2.COLOR_BGR2RGB),cv2.cvtColor(Rr_img,cv2.COLOR_BGR2RGB)])
ax.imshow(pair); ax.axis("off")
for y in np.linspace(80,H-60,9).astype(int): ax.axhline(y,color="#ffd400",lw=0.8,alpha=0.85)
# ponto correspondente exemplo
ys,xs=np.where(valid); k=np.argmin((ys-H*0.55)**2+(xs-W*0.5)**2+1e6*(~valid[ys,xs])) if False else None
yp=0;xp=0
cand=okc&(np.abs(d)<2)&(xx>300)&(xx<1300)&(yy>250)&(yy<650)
ys_,xs_=np.where(cand); k=len(ys_)//2; yp,xp=ys_[k],xs_[k]; dp=disp[yp,xp]  # exemplo concordante
ax.plot([xp,xp-dp+W],[yp,yp],"o",color="red",ms=0)
ax.plot(xp,yp,"o",mfc="none",mec="red",ms=10,mew=2); ax.plot(xp-dp+W,yp,"o",mfc="none",mec="red",ms=10,mew=2)
ax.annotate("",xy=(xp-dp+W,yp),xytext=(xp,yp),arrowprops=dict(arrowstyle="->",color="red",lw=1.5))
ax.text(xp,yp-25,"d = %.0f px"%dp,color="red",fontsize=11,fontweight="bold",bbox=dict(fc="white",ec="none",alpha=.8,pad=1))
# 2 disparidade
ax=fig.add_subplot(gs[0,2]); title(ax,2,"Disparidade (SGBM, px)")
dm=np.where(valid,disp,np.nan); vmax=np.nanpercentile(dm,98)
imd=ax.imshow(dm,cmap="magma",vmin=0,vmax=vmax); ax.axis("off")
cb=plt.colorbar(imd,ax=ax,fraction=0.04,pad=0.01); cb.set_label("d (px)")
ax.text(0.02,0.04,"perto = d grande\nlonge = d pequeno",transform=ax.transAxes,color="white",fontsize=10,bbox=dict(fc="black",alpha=.5,ec="none"))
# 3 formula
ax=fig.add_subplot(gs[1,0]); title(ax,3,"Relação de profundidade")
ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.text(0.5,0.80,r"$Z=\dfrac{f\cdot B}{d}$",ha="center",va="center",fontsize=34)
ax.text(0.5,0.50,F(r"$f=%.1f$ px,   $B=%.1f$ cm",f,Bcm,m=True),ha="center",fontsize=16)
ax.text(0.5,0.36,F(r"$Z=\dfrac{%.1f\cdot %.1f}{d}\approx\dfrac{%d}{d}$ cm",f,Bcm,round(f*Bcm,-1),m=True),ha="center",va="center",fontsize=16)
ax.text(0.5,0.12,"Exemplo: d = %.0f px  ->  Z = %.0f cm"%(dp,f*Bcm/dp),ha="center",fontsize=14,color="red")
ax.text(0.5,-0.03,"f do modelo COLMAP (PINHOLE, imagem 1600x896);\nB = |t| relativo x 29,3745 cm/unidade",ha="center",va="top",fontsize=9,color="#444")
# 4 triangulacao (top-down, retificado)
ax=fig.add_subplot(gs[1,1]); title(ax,4,"Triangulação (vista de cima)")
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.2,2.2); ax.set_ylim(-0.7,3.4)
CL=np.array([0,0]); CR=np.array([1.0,0]); Pp=np.array([0.62,2.6])
for c,lab in ((CL,"$C_E$ (img_1150)"),(CR,"$C_D$ (img_1154)")):
    ax.plot(*c,"s",color="#222",ms=11); 
ax.text(0,-0.1,"$C_E$",ha="center",va="top",fontsize=13); ax.text(1,-0.1,"$C_D$",ha="center",va="top",fontsize=13)
ax.plot([CL[0],Pp[0]],[CL[1],Pp[1]],color="#1f77b4",lw=2); ax.plot([CR[0],Pp[0]],[CR[1],Pp[1]],color="#ff7f0e",lw=2)
ax.plot(*Pp,"o",color="#c00",ms=11); ax.text(Pp[0]+0.07,Pp[1]+0.05,"P (X, Y, Z)",fontsize=13)
ax.annotate("",xy=(1,-0.35),xytext=(0,-0.35),arrowprops=dict(arrowstyle="<->")); 
ax.text(0.5,-0.5,F("B = %.1f cm",Bcm),ha="center",va="top",fontsize=12,fontweight="bold",clip_on=False)
ax.plot([Pp[0],Pp[0]],[0,Pp[1]],"--",color="gray"); ax.text(Pp[0]-0.08,1.3,"Z",ha="right",fontsize=14)
for c,col in ((CL,"#1f77b4"),(CR,"#ff7f0e")):
    ax.plot([c[0]-0.35,c[0]+0.35],[0.55,0.55],color=col,lw=3)
ax.text(-0.35,0.62,"plano da imagem",fontsize=9,color="#444")
ax.text(0.5,3.25,"d = x$_E$ - x$_D$ em px;  f = %.0f px"%f,ha="center",fontsize=11)
# 5 profundidade estereo
vz=np.nanpercentile(pm[pm>0],[2,98])
ax=fig.add_subplot(gs[1,1+1]); title(ax,5,"Profundidade estéreo Z = f·B/d (cm)")
imz=ax.imshow(zshow,cmap="turbo_r" if False else "viridis_r",vmin=vz[0],vmax=vz[1]); ax.axis("off")
cb=plt.colorbar(imz,ax=ax,fraction=0.04,pad=0.01); cb.set_label("Z (cm)")
# 6 comparacao
vmin,vmax2=vz
a1=fig.add_subplot(gs[2,0]); title(a1,6,"PatchMatch (mesma imagem, cm)")
pmv=np.where(pm>0,pm,np.nan); i1=a1.imshow(pmv,cmap="viridis_r",vmin=vmin,vmax=vmax2); a1.axis("off")
plt.colorbar(i1,ax=a1,fraction=0.04,pad=0.01).set_label("Z (cm)")
a2=fig.add_subplot(gs[2,1]); a2.set_title("Diferença estéreo - PatchMatch (cm)",loc="left",fontsize=13,fontweight="bold",color="#1a3c8f")
lim=max(10,np.nanpercentile(np.abs(d),85))
i2=a2.imshow(d,cmap="RdBu",vmin=-lim,vmax=lim); a2.axis("off")
plt.colorbar(i2,ax=a2,fraction=0.04,pad=0.01).set_label("Δ Z (cm)")
a3=fig.add_subplot(gs[2,2]); a3.set_title("Distribuição da diferença",loc="left",fontsize=13,fontweight="bold",color="#1a3c8f")
dd=d[np.isfinite(d)]; a3.hist(dd,bins=np.linspace(*np.percentile(dd,[1,99]),100),color="#4a6fb5")
a3.axvline(0,color="k",lw=0.8); a3.axvline(med,color="red",lw=2)
a3.set_xlabel("Z estéreo - Z PatchMatch (cm)"); a3.spines[["top","right"]].set_visible(False)
a3.text(0.97,0.95,F("mediana = %+.1f cm\n|Δ| mediano = %.1f cm\nerro relativo mediano = %.1f%%\nN = %d px",med,medabs,rel,n),
  transform=a3.transAxes,ha="right",va="top",fontsize=11,bbox=dict(fc="white",ec="#999"))
fig.suptitle("Estéreo da cozinha: par retificado, disparidade, Z = f·B/d e comparação com o PatchMatch",fontsize=15,fontweight="bold",y=0.995)
for a in fig.axes:
    for axis in (a.xaxis,a.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v,_:("%g"%v).replace(".",",").replace("-","\u2212")))
fig.savefig(OUT,dpi=200); print("ok")
json.dump(dict(f=float(f),B=float(Bcm),med=float(med),medabs=float(medabs),rel=float(rel),n=n),open("/tmp/g5.json","w"))
