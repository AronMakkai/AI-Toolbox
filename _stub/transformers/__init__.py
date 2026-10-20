"""Fake SAM 3 with the documented transformers API shapes. The 'model'
finds the brightest blob near the prompt and, in video mode, keeps a
memory of where it was so it follows it frame to frame."""
import numpy as np, json, os
LOG=os.environ.get("FAKE_SAM_LOG")
def _log(ev):
    if LOG:
        with open(LOG,"a") as f: f.write(json.dumps(ev)+"\n")
class _Arr(np.ndarray):
    def to(self,*a,**k): return self
    def tolist(self): return np.asarray(self).tolist()
def _wrap(a): return np.asarray(a).view(_Arr)
class _Batch(dict):
    def to(self,*a,**k): return self
    def __getattr__(self,k): return self[k]
class _Out:
    def __init__(self,**k): self.__dict__.update(k)
def _blob_mask(img, cx, cy, r=None):
    """Mask of the bright blob containing (cx,cy): flood of pixels > 128 near it."""
    import cv2
    g=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2GRAY)
    bw=(g>128).astype(np.uint8)
    n,lab=cv2.connectedComponents(bw)
    cx=int(np.clip(cx,0,g.shape[1]-1)); cy=int(np.clip(cy,0,g.shape[0]-1))
    l=lab[cy,cx]
    if l==0:
        # nearest blob
        ys,xs=np.nonzero(bw)
        if len(xs)==0: return np.zeros(g.shape,np.float32)
        d=(xs-cx)**2+(ys-cy)**2; i=int(np.argmin(d)); l=lab[ys[i],xs[i]]
    return (lab==l).astype(np.float32)
# ---- detector
class Sam3Model:
    @classmethod
    def from_pretrained(cls,*a,**k): return cls()
    def to(self,d): return self
    def eval(self): return self
    def __call__(self, **inputs):
        img=inputs["_img"]
        import cv2
        g=cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2GRAY); bw=(g>128).astype(np.uint8)
        n,lab,stats,cent=cv2.connectedComponentsWithStats(bw,8)
        masks=[(lab==i) for i in range(1,n)]; boxes=[]
        for i in range(1,n):
            x,y,w,h,_=stats[i]; boxes.append([x,y,x+w,y+h])
        return _Out(masks=masks,boxes=boxes)
class Sam3Processor:
    @classmethod
    def from_pretrained(cls,*a,**k): return cls()
    def __call__(self, images=None, text=None, return_tensors=None):
        return _Batch(_img=images, original_sizes=_wrap([[images.height, images.width]]))
    def post_process_instance_segmentation(self, outputs, threshold, mask_threshold, target_sizes):
        return [dict(masks=[_wrap(m) for m in outputs.masks], boxes=[_wrap(b) for b in outputs.boxes],
                     scores=[_wrap(0.9) for _ in outputs.masks])]
# ---- per-frame tracker
class Sam3TrackerProcessor(Sam3Processor):
    def __call__(self, images=None, return_tensors=None, input_points=None, input_labels=None, input_boxes=None):
        return _Batch(_img=images, _pts=input_points, _box=input_boxes)
class Sam3TrackerModel(Sam3Model):
    def __call__(self, **inputs):
        img=inputs["_img"]
        if inputs.get("_pts"): cx,cy=inputs["_pts"][0][0][0]
        else: b=inputs["_box"][0][0]; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        m=_blob_mask(img,cx,cy)
        logit=np.where(m>0, 8.0, -8.0).astype(np.float32)
        return _Out(iou_scores=_wrap([[[0.9]]]), pred_masks=_wrap([[[logit]]]))
# ---- video tracker (streaming)
class _Session:
    def __init__(self): self.prompt=None; self.last_c=None; self.n=0
class Sam3TrackerVideoProcessor:
    @classmethod
    def from_pretrained(cls,*a,**k): return cls()
    def init_video_session(self, inference_device="cpu", **k):
        _log({"ev":"init_session"}); return _Session()
    def __call__(self, images=None, device=None, return_tensors=None):
        return _Batch(pixel_values=[images], original_sizes=_wrap([[images.height, images.width]]))
    def add_inputs_to_inference_session(self, inference_session, frame_idx, obj_ids, input_points=None,
                                        input_labels=None, input_boxes=None, original_size=None):
        _log({"ev":"prompt","frame_idx":frame_idx,"obj_ids":obj_ids,"points":input_points,
              "labels":input_labels,"boxes":input_boxes,"original_size":np.asarray(original_size).tolist()})
        if input_points: cx,cy=input_points[0][0][0]
        else: b=input_boxes[0][0]; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        inference_session.prompt=(cx,cy); inference_session.last_c=(cx,cy)
    def post_process_masks(self, masks, original_sizes, binarize=False):
        return [masks[0]]
class Sam3TrackerVideoModel:
    device="cpu"
    @classmethod
    def from_pretrained(cls,*a,**k): return cls()
    def to(self,d): return self
    def eval(self): return self
    def __call__(self, inference_session=None, frame=None, **k):
        s=inference_session; img=frame
        if s.prompt is None: raise RuntimeError("frame given before any prompt")
        cx,cy=s.last_c
        m=_blob_mask(img,cx,cy)
        ys,xs=np.nonzero(m)
        if len(xs): s.last_c=(float(xs.mean()),float(ys.mean()))   # memory: follow the blob
        s.n+=1
        logit=np.where(m>0, 8.0, -8.0).astype(np.float32)
        return _Out(pred_masks=_wrap([[logit]]), frame_idx=s.n-1)
