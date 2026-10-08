
from pathlib import Path
import json, math, cv2, numpy as np, torch
from ultralytics import YOLO
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4d_format import V4DHeader, write_v4d

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "videos"
OUT = ROOT / "exports" / "reconstructed_color.v4d"
MASK_ROOT = ROOT / "data" / "real_masks"

CAM_NAMES = ["cam0", "cam1", "cam2"]
STEP = 2
GRID = (48, 48, 72)
FPS_OUT = 23.93617021276596 / STEP
BOUNDS_MIN = np.array([-1.25, -1.25, 0.0], dtype=np.float64)
BOUNDS_MAX = np.array([ 1.25,  1.25, 2.25], dtype=np.float64)

def look_at(position, target=np.array([0.0,0.0,1.1])):
    position=np.asarray(position,float); target=np.asarray(target,float)
    forward=target-position; forward/=np.linalg.norm(forward)
    upw=np.array([0.,0.,1.])
    right=np.cross(forward,upw)
    if np.linalg.norm(right)<1e-6: right=np.array([1.,0.,0.])
    right/=np.linalg.norm(right)
    up=np.cross(right,forward); up/=np.linalg.norm(up)
    R=np.vstack([right,up,forward])
    t=-R@position.reshape(3,1)
    return R,t

def camera_for(name,w,h):
    # Approximate circular studio layout inferred from the supplied views:
    # cam0 front, cam1 side, cam2 rear.
    pos={"cam0":np.array([0.,-4.2,1.65]),
         "cam1":np.array([4.2,0.,1.65]),
         "cam2":np.array([0.,4.2,1.65])}[name]
    R,t=look_at(pos)
    f=0.95*max(w,h)
    K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1]],float)
    return {"K":K,"R":R,"t":t,"w":w,"h":h}

def project(points,cam):
    p=(cam["R"]@points.T)+cam["t"]
    z=p[2]
    safe=np.where(np.abs(z)<1e-8,1e-8,z)
    u=cam["K"][0,0]*p[0]/safe+cam["K"][0,2]
    v=cam["K"][1,1]*p[1]/safe+cam["K"][1,2]
    return np.column_stack([u,v]),z

def read_frame(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES,index)
    ok,fr=cap.read()
    return fr if ok else None

def person_mask(model, frame):
    r=model(frame, imgsz=640, conf=0.25, classes=[0], verbose=False)[0]
    if r.masks is None or len(r.masks.data)==0:
        return np.zeros(frame.shape[:2],np.uint8)
    masks=r.masks.data.cpu().numpy()
    boxes=r.boxes
    areas=[]
    for i in range(len(masks)):
        areas.append(float((masks[i] > 0.5).sum()))
    j=int(np.argmax(areas))
    m=(masks[j] > 0.5).astype(np.uint8)*255
    m=cv2.resize(m,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST)
    k=np.ones((5,5),np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,k,iterations=2)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,k,iterations=1)
    return m

def carve(points,cams,masks):
    keep=np.ones(len(points),bool)
    for name,cam in cams.items():
        uv,z=project(points,cam)
        u=np.rint(uv[:,0]).astype(np.int32); v=np.rint(uv[:,1]).astype(np.int32)
        inside=(z>0)&(u>=0)&(u<cam["w"])&(v>=0)&(v<cam["h"])
        visible=np.zeros(len(points),bool)
        idx=np.where(inside)[0]
        if len(idx):
            visible[idx]=masks[name][v[idx],u[idx]]>0
        keep &= visible
        if not keep.any(): break
    return points[keep]

def voxel_grid():
    gx,gy,gz=GRID
    xs=np.linspace(BOUNDS_MIN[0],BOUNDS_MAX[0],gx)
    ys=np.linspace(BOUNDS_MIN[1],BOUNDS_MAX[1],gy)
    zs=np.linspace(BOUNDS_MIN[2],BOUNDS_MAX[2],gz)
    xx,yy,zz=np.meshgrid(xs,ys,zs,indexing="ij")
    return np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])

def color_voxels(points,cams,frames):
    sums=np.zeros((len(points),3),float)
    counts=np.zeros(len(points),np.int16)
    for name,cam in cams.items():
        img=frames[name]
        uv,z=project(points,cam)
        u=np.rint(uv[:,0]).astype(int); v=np.rint(uv[:,1]).astype(int)
        valid=(z>0)&(u>=0)&(u<cam["w"])&(v>=0)&(v<cam["h"])
        idx=np.where(valid)[0]
        if len(idx):
            pix=img[v[idx],u[idx]]
            # Ignore near-background gray studio pixels.
            b,g,r=pix[:,0],pix[:,1],pix[:,2]
            fg=(np.maximum.reduce([b,g,r])-np.minimum.reduce([b,g,r])>12) | (r<145)
            idx=idx[fg]
            if len(idx):
                sums[idx]+=img[v[idx],u[idx]][:,::-1]
                counts[idx]+=1
    rgb=np.zeros((len(points),3),np.uint8)
    good=counts>0
    rgb[good]=np.rint(sums[good]/counts[good,None]).clip(0,255).astype(np.uint8)
    rgb[~good]=np.array([170,170,170],np.uint8)
    return rgb

def main():
    print("REAL 4D HUMAN RECONSTRUCTION")
    print("Loading YOLO segmentation...")
    model=YOLO("yolo11n-seg.pt")
    device=0 if torch.cuda.is_available() else "cpu"
    print("Device:", "RTX/CUDA" if device==0 else "CPU")
    caps={}
    meta={}
    for name in CAM_NAMES:
        p=VIDEOS/(name+".mp4")
        cap=cv2.VideoCapture(str(p))
        if not cap.isOpened(): raise SystemExit(f"Missing video: {p}")
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
        caps[name]=cap
        meta[name]=(n,fps,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total=min(v[0] for v in meta.values())
    fps=min(v[1] for v in meta.values())
    indices=list(range(0,total,STEP))
    cameras={name:camera_for(name,meta[name][2],meta[name][3]) for name in CAM_NAMES}
    grid=voxel_grid()
    frames_out=[]
    MASK_ROOT.mkdir(parents=True,exist_ok=True)
    for out_i,src_i in enumerate(indices):
        masks={}; rgbframes={}
        for name in CAM_NAMES:
            fr=read_frame(caps[name],src_i)
            if fr is None: raise SystemExit(f"Failed frame {src_i} in {name}")
            rgbframes[name]=fr
            mp=MASK_ROOT/name/f"{src_i:04d}.png"; mp.parent.mkdir(parents=True,exist_ok=True)
            if mp.exists():
                m=cv2.imread(str(mp),cv2.IMREAD_GRAYSCALE)
            else:
                m=person_mask(model,fr)
                cv2.imwrite(str(mp),m)
            masks[name]=m
        pts=carve(grid,cameras,masks)
        if len(pts):
            rgb=color_voxels(pts,cameras,rgbframes)
            norm=(pts-BOUNDS_MIN)/(BOUNDS_MAX-BOUNDS_MIN)
            coords=np.rint(norm*(np.asarray(GRID)-1)).astype(np.int32)
            coords=np.clip(coords,0,np.asarray(GRID)-1)
            vox=[(int(x),int(y),int(z),int(r),int(g),int(b)) for (x,y,z),(r,g,b) in zip(coords,rgb)]
        else:
            vox=[]
        ts=src_i/fps
        frames_out.append((ts,vox))
        print(f"{out_i+1:03d}/{len(indices)}  t={ts:5.2f}s  voxels={len(vox)}")
    for c in caps.values(): c.release()
    header=V4DHeader(GRID[0],GRID[1],GRID[2],len(frames_out),fps/STEP,float(np.mean((BOUNDS_MAX-BOUNDS_MIN)/(np.asarray(GRID)-1))))
    write_v4d(OUT,header,frames_out)
    import shutil
    (ROOT/"player"/"public").mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, ROOT/"player"/"public"/"reconstructed_color.v4d")
    print("DONE:",OUT)
    print("Frames:",len(frames_out),"FPS:",fps/STEP)

if __name__=="__main__":
    main()
