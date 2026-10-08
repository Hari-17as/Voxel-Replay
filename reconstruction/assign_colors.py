"""Assign RGB only from foreground pixels seen by reconstruction cameras."""
from pathlib import Path
import json, cv2, numpy as np, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from silhouette_carving import load_cameras, project_points
from v4d_format import read_all_frames, write_v4d

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT/"data"/"capture"
INPUT = ROOT/"exports"/"reconstructed.v4d"
OUTPUT = ROOT/"exports"/"reconstructed_color.v4d"
USE = ["cam0","cam1","cam2"]

def world(v,h,bmin,bmax):
    return np.array([
        bmin[0]+v[0]/max(h.grid_x-1,1)*(bmax[0]-bmin[0]),
        bmin[1]+v[1]/max(h.grid_y-1,1)*(bmax[1]-bmin[1]),
        bmin[2]+v[2]/max(h.grid_z-1,1)*(bmax[2]-bmin[2])
    ], dtype=np.float64)

meta=json.loads((CAP/"capture.json").read_text())
cams_all=load_cameras(CAP/"cameras.json")
cams={n:cams_all[n] for n in USE}
bmin=np.array(meta["bounds_min"],float); bmax=np.array(meta["bounds_max"],float)
header,frames=read_all_frames(INPUT)
out=[]

for fi,(ts,voxels) in enumerate(frames):
    if not voxels:
        out.append((ts,[])); continue
    pts=np.array([world(v,header,bmin,bmax) for v in voxels])
    sums=np.zeros((len(pts),3),float)
    counts=np.zeros(len(pts),int)

    for name,cam in cams.items():
        rgbp=CAP/"rgb"/name/f"{fi:04d}.png"
        maskp=CAP/"masks"/name/f"{fi:04d}.png"
        img=cv2.imread(str(rgbp),cv2.IMREAD_COLOR)
        mask=cv2.imread(str(maskp),cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        uv,depth=project_points(pts,cam)
        u=np.rint(uv[:,0]).astype(int); v=np.rint(uv[:,1]).astype(int)
        valid=(depth>0)&(u>=0)&(u<cam.image_width)&(v>=0)&(v<cam.image_height)
        idx=np.where(valid)[0]
        if len(idx):
            fg=mask[v[idx],u[idx]]>0
            idx=idx[fg]
            if len(idx):
                sums[idx]+=img[v[idx],u[idx]]
                counts[idx]+=1

    colored=[]
    for i,old in enumerate(voxels):
        if counts[i]:
            rgb=np.rint(sums[i]/counts[i]).astype(np.uint8)
            r,g,b=map(int,rgb)
        else:
            r,g,b=map(int,old[3:6])
        colored.append((int(old[0]),int(old[1]),int(old[2]),r,g,b))
    out.append((ts,colored))
    print(f"frame {fi+1:02d}/{len(frames)}: {len(colored):5d} voxels colored")

write_v4d(OUTPUT,header,out)
print("COLOR ASSIGNMENT FIXED")
print(f"Output: {OUTPUT}")
print("Only foreground-mask pixels were used for RGB.")
