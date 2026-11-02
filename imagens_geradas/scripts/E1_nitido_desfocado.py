import cv2, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/home/moliveira/Documents/3d_indoor_environment/"
d=pd.read_csv(R+"cozinha/artifacts/config_cozinha/frame_quality_metrics.csv").set_index("frame")
med=float(d.laplacian_var.median()); lim=0.5*med
fr={"nitido":2091,"desfocado":2090}
img={k:cv2.imread(R+f"cozinha/frames/all_stride10/img_{v:04d}.png") for k,v in fr.items()}
row={k:d.loc[f"cozinha/frames/all_stride10/img_{v:04d}.png"] for k,v in fr.items()}
crop={"nitido":(1150,520),"desfocado":(1010,580)}; W,H=400,300
sift=cv2.SIFT_create()
kp={}
for k,im in img.items():
    s=cv2.resize(im,(960,540)); kp[k]=sift.detect(cv2.cvtColor(s,cv2.COLOR_BGR2GRAY),None)
fig,ax=plt.subplots(3,2,figsize=(12,12.5),dpi=220)
for c,k in enumerate(fr):
    x,y=crop[k]; im=img[k]
    a=ax[0,c]; a.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); a.add_patch(plt.Rectangle((x,y),W,H,fill=False,ec="red",lw=1.5)); a.axis("off")
    r=row[k]; rot="nítido" if k=="nitido" else "desfocado"
    a.set_title(f"Quadro {fr[k]:04d} ({rot})\nvariância do Laplaciano = {format(r.laplacian_var,'.1f').replace('.',',')}  |  {'mantido' if r.keep else 'marcado: '+str(r.reason)}",fontsize=9)
    b=ax[1,c]; b.imshow(cv2.cvtColor(im[y:y+H,x:x+W],cv2.COLOR_BGR2RGB)); b.axis("off")
    b.set_title("Recorte 1:1 (400×300 px) da textura da cesta",fontsize=9)
    s=cv2.resize(im,(960,540)); vis=cv2.drawKeypoints(s,kp[k],None,color=(0,255,0))
    e=ax[2,c]; e.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)); e.axis('off'); e.set_title(f'Keypoints SIFT: {len(kp[k])}',fontsize=9)
fig.suptitle(f"Quadros vizinhos da cozinha (passo 10): limiar de desfoque = 0,5 × mediana ({format(med,'.1f').replace('.',',')}) = {format(lim,'.1f').replace('.',',')};  keypoints SIFT (quadro reduzido a 960×540): nítido {len(kp['nitido'])}, desfocado {len(kp['desfocado'])}",fontsize=9.5)
fig.tight_layout(); fig.savefig("E1_nitido_desfocado.png",dpi=220)
print(med,lim,{k:(fr[k],row[k].laplacian_var,row[k].keep,row[k].reason,row[k].fast_keypoints,len(kp[k])) for k in fr})
