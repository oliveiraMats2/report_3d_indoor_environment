import cv2, numpy as np, pycolmap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker
R="/home/moliveira/Documents/3d_indoor_environment/"
REC=R+"cozinha/results/config_cozinha/exports/colmap/sparse/0"
FR=R+"cozinha/frames/all_stride10/img_1320.png"
OUT="/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/G4_distorcao_correcao.png"
N=8  # fator de amplificacao (so visualizacao)
cam=list(pycolmap.Reconstruction(REC).cameras.values())[0]
fx,fy,cx,cy,k1,k2,p1,p2=cam.params; W,H=cam.width,cam.height
K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]); D=np.array([k1,k2,p1,p2,0.])
img=cv2.cvtColor(cv2.imread(FR),cv2.COLOR_BGR2RGB)
und=cv2.undistort(img,K,D,None,K)

def distort_pts(u,v,s=1.0):
    x=(u-cx)/fx; y=(v-cy)/fy; r2=x*x+y*y
    a,b,c,d=k1*s,k2*s,p1*s,p2*s
    rad=1+a*r2+b*r2*r2
    xd=x*rad+2*c*x*y+d*(r2+2*x*x); yd=y*rad+c*(r2+2*y*y)+2*d*x*y
    return xd*fx+cx, yd*fy+cy
uu,vv=np.meshgrid(np.arange(W),np.arange(H)); uu=uu.astype(np.float64); vv=vv.astype(np.float64)
xd,yd=distort_pts(uu,vv); mag=np.hypot(xd-uu,yd-vv)
mx=mag.max(); iy,ix=np.unravel_index(mag.argmax(),mag.shape); print("pos max",ix,iy); mc=mag[0,0]; mm=mag[0,W//2]; ml=mag[H//2,0]
print("max px",mx,"canto",mc,"meio borda sup",mm,"meio borda esq",ml,"centro",mag[H//2,W//2])
# imagem amplificada: aplica a distorcao com coeficientes xN sobre a imagem corrigida
xa,ya=distort_pts(uu,vv,N)
amp=cv2.remap(und,xa.astype(np.float32),ya.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
def grid(ax,step=160):
    for x in range(0,W+1,step): ax.axvline(x,color="#ff2020",lw=.7,alpha=.9)
    for y in range(0,H+1,step): ax.axhline(y,color="#ff2020",lw=.7,alpha=.9)
fig=plt.figure(figsize=(16,9),dpi=220)
def AX(c,r,w=.30,h=None,dx=0):
    return fig.add_axes([.03+c*.325,[.56,.10][r],.30,.30])
def show(ax,im,t):
    ax.imshow(im,extent=(0,W,H,0)); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.set_title(t,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
ax=AX(0,0); show(ax,img,"(1) Quadro original (img_1320) com grade regular"); grid(ax)
ax=AX(1,0); show(ax,img,"(2) Campo de deslocamento da distorção")
m=ax.imshow(mag,extent=(0,W,H,0),cmap="magma",alpha=.78,vmin=0,vmax=mx)
st=120; qu,qv=np.meshgrid(np.arange(st//2,W,st),np.arange(st//2,H,st))
qx,qy=distort_pts(qu.astype(float),qv.astype(float))
ax.quiver(qu,qv,(qx-qu)*N,(qy-qv)*N,color="cyan",angles="xy",scale_units="xy",scale=1,width=.003)
ax.set_xlim(0,W); ax.set_ylim(H,0)
cb=fig.colorbar(m,cax=fig.add_axes([.355,.50,.30,.012]),orientation="horizontal"); cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,_:f"{v:g}".replace(".",",")))
cb.set_label("|deslocamento| (px)",fontsize=9)
ax.text(10,H-20,f"setas amplificadas ×{N}, só para visualização",color="cyan",fontsize=9,backgroundcolor="#000a")
ax=AX(2,0); show(ax,und,"(3) Quadro corrigido (cv2.undistort, mesmo K) e grade"); grid(ax)
# zoom: canto superior esquerdo
x0,x1,y0,y1=0,360,330,740
def zoom(ax,im,t):
    ax.imshow(im[y0:y1,x0:x1],extent=(x0,x1,y1,y0)); ax.set_title(t,fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax.axvline(160,color="#ff2020",lw=1); ax.axhline(480,color="#ff2020",lw=1)
    ax.axvline(40,color="#00e000",lw=1,ls="--")
zoom(fig.add_axes([.03,.10,.148,.30]),img,"(4a) Zoom na borda esquerda: antes")
zoom(fig.add_axes([.182+.0,.10,.148,.30]),und,"(4b) depois")
ax=AX(1,1); show(ax,amp,f"(5) Distorção amplificada ×{N}, só para visualização"); grid(ax)
ax.text(10,H-20,f"coeficientes k1,k2,p1,p2 multiplicados por {N}",color="white",fontsize=9,backgroundcolor="#000a")
ax=fig.add_axes([.68,.10,.31,.30]); ax.axis("off")
txt=(f"Intrínsecos reais do SfM (OPENCV {W}x{H})\n"
f"fx = {fx:.3f}   fy = {fy:.3f}\ncx = {cx:.0f}   cy = {cy:.0f}\n\n"
"K = [ fx  0  cx ]\n    [ 0  fy  cy ]\n    [ 0   0   1 ]\n\n"
f"k1 = {k1:.5f}\nk2 = {k2:.5f}\np1 = {p1:.5f}\np2 = {p2:.1e}\n\n"
"Deslocamento |distorcido - ideal| (px):\n"
f"  centro        {mag[H//2,W//2]:.2f}\n  meio da borda esq. {ml:.1f}\n  meio da borda sup. {mm:.1f}\n  máximo ({ix},{iy}) {mx:.1f}\n  canto sup.-esq.   {mc:.1f}\n\n"
"Modelo: x_d = x(1+k1 r²+k2 r⁴) + tangencial\n"
"Fontes: cameras.bin (sparse/0) e\ncozinha/frames/all_stride10/img_1320.png")
import re; txt=re.sub(r"(?<=\d)\.(?=\d)",",",txt)
ax.text(0,1,txt,va="top",family="monospace",fontsize=7.6,bbox=dict(boxstyle="round",fc="#f4f4f4",ec="#888"))
fig.savefig(OUT,bbox_inches="tight"); 
