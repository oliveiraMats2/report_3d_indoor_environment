"""G15: wireframe da malha limpa da cozinha (decimada) + zoom na malha densa original.
Fonte: 3d_indoor_environment/cozinha/results/config_cozinha/malha_limpa/malha_limpa.ply (resumo.json: 3.444.576 vert., 6.888.396 faces)."""
import open3d as o3d, open3d.visualization.rendering as rd, numpy as np, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/home/moliveira/Documents/3d_indoor_environment"
PLY=R+"/cozinha/results/config_cozinha/malha_limpa/malha_limpa.ply"
SCR="/tmp/claude-1009/-home-moliveira-Documents-report-3d-indoor-environment/9dba2e02-911f-44c4-b590-08bb53b5d3f4/scratchpad"
OUT="/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/G15_wireframe_malha.png"
NF=40000
dense=o3d.io.read_triangle_mesh(PLY)
dp=SCR+"/dec40k.ply"
if os.path.exists(dp): dec=o3d.io.read_triangle_mesh(dp)
else:
    dec=dense.simplify_quadric_decimation(NF); dec.remove_degenerate_triangles(); dec.remove_unreferenced_vertices(); o3d.io.write_triangle_mesh(dp,dec)
UP=np.array([0,0,1.])
T=np.load(R+"/cozinha/results/config_cozinha/mesh/room_fill/T_room.npy")
Ti=np.linalg.inv(T)
dense.transform(Ti); dec.transform(Ti)   # referencial da sala (x,y,h em cm)

def wire(m, eps):
    V=np.asarray(m.vertices); F=np.asarray(m.triangles)
    E=np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
    return V,F,E

def render(m, eye, center, W, H, lw, fov=45, eps=0.3, ret_nf=False):
    V,F,E=wire(m,eps)
    base=o3d.geometry.TriangleMesh(m.vertices,m.triangles); base.paint_uniform_color([1,1,1])
    # linhas deslocadas levemente em direcao a camera para evitar z-fighting
    d=eye-V; d/=np.linalg.norm(d,axis=1,keepdims=True)
    ls=o3d.geometry.LineSet(o3d.utility.Vector3dVector(V+eps*d),o3d.utility.Vector2iVector(E))
    ls.paint_uniform_color([0.10,0.20,0.30])
    r=rd.OffscreenRenderer(W,H); s=r.scene; s.set_background([1,1,1,1])
    mm=rd.MaterialRecord(); mm.shader="defaultUnlit"; mm.base_color=[1,1,1,1]
    ml=rd.MaterialRecord(); ml.shader="unlitLine"; ml.line_width=lw; ml.base_color=[0.10,0.20,0.30,1]
    s.add_geometry("m",base,mm); s.add_geometry("l",ls,ml)
    r.setup_camera(fov,center,eye,UP)
    for _ in range(4):
        img=np.asarray(r.render_to_image())
        if img.sum()>0 and _>=1: break
    del r
    return img.copy(), len(F)

def crop(m, lo, hi):
    bb=o3d.geometry.AxisAlignedBoundingBox(lo,hi)
    return m.crop(bb)

if __name__=="__main__":
    c=(np.asarray(dec.get_min_bound())+np.asarray(dec.get_max_bound()))/2; c[2]=110
    az=np.radians(10); el=np.radians(50); dist=760
    eye=c+dist*np.array([np.sin(az)*np.cos(el), -np.cos(az)*np.cos(el), np.sin(el)])
    # regiao do zoom: celula 45x45x45 cm com mais vertices da malha decimada entre h=70 e 130 cm
    V=np.asarray(dec.vertices); sel=(V[:,2]>70)&(V[:,2]<130)
    H,xe,ye=np.histogram2d(V[sel,0],V[sel,1],bins=[int(300/45),int(460/45)],range=[[-170,130],[-260,200]])
    ix,iy=np.unravel_index(H.argmax(),H.shape)
    lo=np.array([xe[ix],ye[iy],70.]); hi=lo+np.array([45,45,45.]); print("caixa",lo,hi)
    img,nf=render(dec,eye,c,2400,1900,1.0)
    img[(img[...,:3].min(2)>205)]=255
    zoom=crop(dense,lo,hi); print("faces zoom",len(zoom.triangles))
    zc=(lo+hi)/2; zeye=zc+95*np.array([0.25,-0.8,0.55])
    zimg,nz=render(zoom,zeye,zc,1800,1800,1.0,fov=45,eps=0.02)
    zimg[(zimg[...,:3].min(2)>205)]=255
    # caixa do zoom: projetar com matriz de camera aproximada via renderer nao exposta; usar LineSet separado
    np.save(SCR+"/img.npy",img); np.save(SCR+"/zimg.npy",zimg)
    import json; json.dump(dict(nf=nf,nz=len(zoom.triangles),nvz=len(zoom.vertices),lo=lo.tolist(),hi=hi.tolist(),nv=len(dec.vertices)),open(SCR+"/info.json","w"))
    plt.imsave(SCR+"/full.png",img); plt.imsave(SCR+"/zoom.png",zimg)
    # ---- composicao final ----
    def proj(P, eye, ctr, W, H, fov=45):
        f=ctr-eye; f/=np.linalg.norm(f); r_=np.cross(f,UP); r_/=np.linalg.norm(r_); u=np.cross(r_,f)
        d=P-eye; x=d@r_; y=d@u; z=d@f; fl=(H/2)/np.tan(np.radians(fov/2))
        return W/2+fl*x/z, H/2-fl*y/z
    corners=np.array([[a,b,c_] for a in (lo[0],hi[0]) for b in (lo[1],hi[1]) for c_ in (lo[2],hi[2])])
    px,py=proj(corners,eye,c,2400,1900)
    nz_=lambda a: np.where((a<255).any(2))
    ys,xs=nz_(img[...,:3]); x0,x1,y0,y1=xs.min()-40,xs.max()+40,ys.min()-40,ys.max()+40
    fig=plt.figure(figsize=(16,8.2),dpi=220)
    ax=fig.add_axes([0.01,0.07,0.55,0.86]); ax.imshow(img[y0:y1,x0:x1]); ax.axis("off")
    bx=[px.min()-x0,px.max()-x0,px.max()-x0,px.min()-x0,px.min()-x0]; by=[py.min()-y0,py.min()-y0,py.max()-y0,py.max()-y0,py.min()-y0]
    ax.plot(bx,by,color="#c0392b",lw=2)
    ax.set_title("(a) Malha limpa decimada: %d faces, %d vértices (vista oblíqua, sem teto)"%(nf,len(dec.vertices)),fontsize=13)
    ax2=fig.add_axes([0.58,0.07,0.41,0.86]); ax2.imshow(zimg); ax2.axis("off")
    for s_ in ax2.spines.values(): pass
    ax2.set_title("(b) Região em vermelho em (a), malha densa original:\n%d faces (caixa de 45 cm de lado)"%len(zoom.triangles),fontsize=13)
    fig.text(0.5,0.015,"Malha limpa da cozinha (Poisson): 3.444.576 vértices e 6.888.396 faces; (a) reduzida por quadric decimation (Open3D) a %d faces; (b) sem simplificação. Furos = regiões sem superfície reconstruída."%nf,ha="center",fontsize=11)
    fig.savefig(OUT,dpi=220); print("OK",OUT)
