"""Fig. fig:kp: keypoints SIFT, ORB e AKAZE no quadro img_1110 (mesmos parametros de src/detectores.py)."""
import csv, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "/home/moliveira/Documents/3d_indoor_environment"
ART = f"{R}/cozinha/artifacts/config_cozinha"
OUT = "/home/moliveira/Documents/report_3d_indoor_environment/images/report/keypoints.jpg"
img = cv2.imread(f"{R}/cozinha/frames/all_stride10/img_1110.png")
gray = cv2.imread(f"{R}/cozinha/frames/all_stride10/img_1110.png", cv2.IMREAD_GRAYSCALE)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
N = 5000
dets = {"SIFT": cv2.SIFT_create(nfeatures=N), "ORB": cv2.ORB_create(nfeatures=N),
        "AKAZE": cv2.xfeatures2d.AKAZE_create(max_points=N)}
ch = {r["detector"]: r for r in csv.DictReader(open(f"{ART}/detector_choice.csv"))}
dm = {r["detector"]: int(r["n_keypoints"]) for r in csv.DictReader(open(f"{ART}/detection_metrics.csv")) if r["frame"] == "img_1110"}
fig, axs = plt.subplots(1, 3, figsize=(10, 2.45), dpi=150)
for ax, (nome, det) in zip(axs, dets.items()):
    kp = det.detect(gray, None)
    assert len(kp) == dm[nome.lower()], (nome, len(kp))
    ax.imshow(rgb)
    ax.scatter([k.pt[0] for k in kp], [k.pt[1] for k in kp], s=3.5, c="#00ff00", linewidths=0)
    r = ch[nome.lower()]
    ax.set_title(f"{nome}: {len(kp)} pontos\nescore {float(r['score']):.2f} | {float(r['ransac_inliers']):.1f} inliers/par".replace('.', ','),
                 fontsize=9)
    ax.axis("off")
plt.subplots_adjust(left=0.003, right=0.997, top=0.80, bottom=0.01, wspace=0.01)
fig.savefig(OUT, format="jpg", pil_kwargs={"quality": 80})
