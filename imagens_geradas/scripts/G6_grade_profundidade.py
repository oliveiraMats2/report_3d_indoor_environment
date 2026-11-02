import numpy as np, cv2, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm, colors
R='/home/moliveira/Documents/3d_indoor_environment/cozinha/results/config_cozinha/dense/'
ESC=29.3745  # cm/unidade
FR=[(600,'Bancada e armário'),(960,'Geladeira'),(1200,'Pia'),(2280,'Fogão (tampa)')]
def rb(p):
    with open(p,'rb') as f:
        w,h,c=[int(x) for x in b''.join(iter(lambda:f.read(1),b'&')) .split(b'&')[:0]] if False else (0,0,0)
        hdr=[];n=0
        while n<3:
            ch=f.read(1)
            if ch==b'&': n+=1
            hdr.append(ch)
        w,h,c=[int(x) for x in b''.join(hdr).split(b'&')[:3]]
        return np.fromfile(f,np.float32).reshape(c,h,w).transpose(1,2,0)[...,0] if c>1 else np.fromfile(f,np.float32).reshape(h,w)
fig,ax=plt.subplots(len(FR),3,figsize=(13,2.6*len(FR)+0.6),dpi=220)
rng=[]
for r,(i,nome) in enumerate(FR):
    img=cv2.cvtColor(cv2.imread(R+f'undistorted/images/img_{i:04d}.png'),cv2.COLOR_BGR2RGB)
    d=rb(R+f'patch_match/ws/stereo/depth_maps/img_{i:04d}.png.geometric.bin')*ESC
    h,w=d.shape; img=cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
    ok=np.isfinite(d)&(d>0)
    lo,hi=np.percentile(d[ok],[2,98]); rng.append((i,d[ok].min(),d[ok].max(),lo,hi,ok.mean()))
    inv=np.zeros_like(d); inv[ok]=1/np.clip(d[ok],lo,hi)
    n=np.zeros_like(d); n[ok]=(inv[ok]-1/hi)/(1/lo-1/hi)
    g=np.zeros((h,w,3)); g[ok]=n[ok,None]
    j=cm.jet(n)[...,:3]; j[~ok]=0
    for c,x in enumerate([img,g,j]): ax[r,c].imshow(x); ax[r,c].axis('off')
    ax[r,0].set_title(f'quadro {i}: {nome}',fontsize=9,loc='left')
    cax=ax[r,2].inset_axes([1.02,0,0.03,1]); cb=fig.colorbar(cm.ScalarMappable(colors.Normalize(0,1),'jet'),cax=cax)
    t=np.linspace(0,1,5); cb.set_ticks(t)
    cb.set_ticklabels([f'{1/(1/hi+v*(1/lo-1/hi)):.0f}' for v in t]); cb.set_label('profundidade (cm)',fontsize=8); cb.ax.tick_params(labelsize=7)
for c,t in enumerate(['Imagem RGB','Profundidade inversa (cinza)','Profundidade inversa (pseudo-cor jet)']): ax[0,c].text(0.5,1.18,t,transform=ax[0,c].transAxes,ha='center',fontsize=11,weight='bold')
plt.tight_layout(rect=(0,0,0.97,1))
plt.savefig('../G6_grade_profundidade.png',dpi=220); print(rng)
