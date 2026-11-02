BOX = (-5.9, -1.9, -3.0, 0.4)  # recorte x_min,x_max,z_min,z_max (m)
# G3: malha LiDAR da cozinha em cinza, sem textura, sem teto (corte por altura).
import sys, numpy as np, open3d as o3d
from open3d.visualization import rendering
from PIL import Image
R = '/home/moliveira/Documents/3d_indoor_environment'
OUT = '/home/moliveira/Documents/report_3d_indoor_environment/imagens_geradas/G3_lidar_malha_cinza.png'
YCUT = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # remove teto: y > YCUT (m)
m = o3d.io.read_triangle_mesh(f'{R}/cozinha_lidar/fused/mesh_closed_2M.ply')
nv, nf = len(m.vertices), len(m.triangles)
m.vertex_colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
v = np.asarray(m.vertices); t = np.asarray(m.triangles)
X0, X1, Z0, Z1 = BOX
inb = (v[:,0]>X0)&(v[:,0]<X1)&(v[:,2]>Z0)&(v[:,2]<Z1)&(v[:,1]<YCUT)&(v[:,1]>-1.05)
keep = inb[t].all(axis=1)
m.remove_triangles_by_mask(~keep); m.remove_unreferenced_vertices()
m.compute_vertex_normals()
print('original', nv, nf, 'recorte', len(m.vertices), len(m.triangles))
mat = rendering.MaterialRecord(); mat.shader = 'defaultLit'
mat.base_color = [0.78, 0.78, 0.78, 1.0]; mat.base_roughness = 0.9; mat.base_reflectance = 0.2

def render(eye, center, up, w=3000, h=2000, fov=60):
    r = rendering.OffscreenRenderer(w, h)
    s = r.scene; s.set_background([1, 1, 1, 1]); s.add_geometry('m', m, mat)
    s.scene.set_sun_light([-0.4, -1, -0.5], [1, 1, 1], 70000)
    s.scene.enable_sun_light(True)
    r.setup_camera(fov, center, eye, up)
    r.render_to_image()
    img = np.asarray(r.render_to_image()).copy()
    bgmask = np.asarray(r.render_to_depth_image()) >= 0.9999   # fundo = profundidade maxima
    del r
    img[bgmask] = 255
    ys, xs = np.where(~bgmask); p = 40
    img = img[max(ys.min()-p,0):ys.max()+p, max(xs.min()-p,0):xs.max()+p]
    return img

b = m.get_axis_aligned_bounding_box(); c = b.get_center()
c[1] = -0.5
img = render(c + np.array([1.0, 5.5, 2.6]), c, [0, 1, 0])
Image.fromarray(img).save(OUT, dpi=(300, 300))
