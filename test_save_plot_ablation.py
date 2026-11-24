# SAME VERSION OF Test_save_PLOT but with the use of NVDIFFRAST
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import argparse, csv, json, math, sys, subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import cv2, hydra, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from hydra import compose, initialize
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from pytorch3d.transforms import so3_exp_map
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from bop_toolkit_lib import inout


def quat_to_mat(q):
    q=q/q.norm(dim=-1,keepdim=True)
    x,y,z,w=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],dim=-1).reshape(*q.shape[:-1],3,3)


def super_fibonacci(n,device='cpu'):
    phi=math.sqrt(2.0); psi=1.533751168755204288118041
    s=torch.arange(n,device=device,dtype=torch.float64)+.5
    r,rr=torch.sqrt(s/n),torch.sqrt(1-s/n)
    a,b=2*math.pi*s/phi,2*math.pi*s/psi
    q=torch.stack([r*torch.sin(a),r*torch.cos(a),rr*torch.sin(b),rr*torch.cos(b)],-1)
    return quat_to_mat(q).float()


def random_rotations(shape,device):
    return quat_to_mat(torch.randn(*shape,4,device=device))


def so3_exp(w):
    return so3_exp_map(w.reshape(-1,3)).reshape(*w.shape[:-1],3,3)


class CnosDetDataset(Dataset):
    def __init__(self,root,cnos_json,targets_json,split='test',crop_size=256,
                 margin=1.2,min_score=0.,min_mask_px=50,det_per_target=3):
        with open(cnos_json) as f: dets=json.load(f)
        with open(targets_json) as f:
            inst={(d['scene_id'],d['im_id'],d['obj_id']):d['inst_count'] for d in json.load(f)}
        groups=defaultdict(list)
        for d in dets:
            k=(d['scene_id'],d['image_id'],d['category_id'])
            if d['score']>=min_score and k in inst: groups[k].append(d)
        sel=[]
        for k,ds in groups.items():
            if det_per_target>0:
                ds.sort(key=lambda d:d['score'],reverse=True)
                ds=ds[:max(inst[k]*det_per_target,1)]
            sel.extend(ds)
        self.dets=sorted(sel,key=lambda d:(d['category_id'],d['scene_id'],d['image_id']))
        print(f'detection CNOS: {len(dets)} -> {len(self.dets)} after filters',flush=True)
        self.root,self.split,self.size,self.margin,self.min_px=root,split,crop_size,margin,min_mask_px
        self._cams={}
    def __len__(self): return len(self.dets)
    def _cam(self,sid):
        if sid not in self._cams:
            with open(f'{self.root}/{self.split}/{sid:06d}/scene_camera.json') as f:
                self._cams[sid]=json.load(f)
        return self._cams[sid]
    def _read_image(self,base,iid):
        for sub,ext in [('rgb','png'),('rgb','jpg'),('gray','tif')]:
            p=f'{base}/{sub}/{iid:06d}.{ext}'
            if os.path.exists(p):
                img=cv2.imread(p,cv2.IMREAD_COLOR)
                if img is not None: return img[...,::-1]
        raise FileNotFoundError(f'No RGB image for {base} / {iid}')
    def guess_translation(self,depth,mask,K):
        ys,xs=np.where(mask>0)
        if not len(xs): return np.zeros(3,np.float32),False
        uc,vc=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2
        valid=(mask>0)&(depth>0)
        if valid.sum()<self.min_px: return np.zeros(3,np.float32),False
        z=float(np.median(depth[valid]))
        return ((np.linalg.inv(K)@np.array([uc,vc,1.]))*z).astype(np.float32),True
    def _get_crop_box(self,mask,bbox):
        ys,xs=np.where(mask)
        if len(xs): x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max()
        else:
            x,y,w,h=bbox; x0,x1,y0,y1=x,x+w,y,y+h
        cx,cy=(x0+x1)/2,(y0+y1)/2
        half=max(max(x1-x0,y1-y0)*self.margin/2,1.)
        return tuple(map(lambda x:int(round(x)),(cx-half,cy-half,cx+half,cy+half)))
    def __getitem__(self,i):
        d=self.dets[i];sid,iid=d['scene_id'],d['image_id']
        base=f'{self.root}/{self.split}/{sid:06d}'
        img=self._read_image(base,iid)
        cam=self._cam(sid)[str(iid)]
        K=np.asarray(cam['cam_K'],dtype=np.float32).reshape(3,3)
        depth=cv2.imread(f'{base}/depth/{iid:06d}.png',cv2.IMREAD_UNCHANGED)
        depth=depth.astype(np.float32)*cam.get('depth_scale',1.)
        rle=dict(d['segmentation']); h,w=rle['size']
        if isinstance(rle['counts'],list): rle=mask_utils.frPyObjects(rle,h,w)
        elif isinstance(rle['counts'],str):rle['counts']=rle['counts'].encode('ascii')
        mask=mask_utils.decode(rle).astype(np.uint8)
        t0,ok=self.guess_translation(depth,mask,K)
        box=self._get_crop_box(mask>0,d['bbox']); xmin,ymin,xmax,ymax=box
        outsz=(self.size,self.size)
        img_c=np.asarray(Image.fromarray(np.ascontiguousarray(img)).crop(box).resize(outsz,Image.Resampling.BICUBIC))
        mask_c=np.asarray(Image.fromarray(mask*255).crop(box).resize(outsz,Image.Resampling.NEAREST))
        dep_c=np.asarray(Image.fromarray(depth).crop(box).resize(outsz,Image.Resampling.NEAREST))
        sx,sy=self.size/float(xmax-xmin),self.size/float(ymax-ymin)
        Kc=K.copy();Kc[0,0]*=sx;Kc[1,1]*=sy
        Kc[0,2]=(K[0,2]-xmin)*sx;Kc[1,2]=(K[1,2]-ymin)*sy
        return {'img':torch.from_numpy(img_c.copy()).permute(2,0,1).float()/255.,
                'mask':torch.from_numpy(mask_c.copy()).float()[None]/255.,
                'depth':torch.from_numpy(dep_c.copy())[None],'K':torch.from_numpy(Kc),
                't0':torch.from_numpy(t0),'valid':torch.tensor(ok),
                'box':torch.tensor(box,dtype=torch.int32),'idx':i,
                'scene_id':sid,'im_id':iid,'obj_id':d['category_id'],'det_score':float(d['score'])}


class NvRenderer:
    """Identical rasterizer pathway to baseline, black background, AA, vertex colors/UV textures."""
    def __init__(self,models_dir,device,size=256,near=1.,far=100000.,allow_vertex_colors=True):
        self.models_dir=Path(models_dir);self.device=torch.device(device)
        self.size=size;self.near=float(near);self.far=float(far)
        self.allow_vertex_colors=allow_vertex_colors
        self.ctx=dr.RasterizeCudaContext(device=self.device);self.cache={}
    def _load(self,obj_id):
        if obj_id in self.cache:return self.cache[obj_id]
        path=self.models_dir/f'obj_{obj_id:06d}.ply'
        if not path.is_file():raise FileNotFoundError(path)
        mesh=trimesh.load(str(path),force='mesh',process=False)
        if not isinstance(mesh,trimesh.Trimesh):raise ValueError(f'Not a triangular mesh: {path}')
        verts=np.asarray(mesh.vertices,dtype=np.float32)
        faces=np.asarray(mesh.faces,dtype=np.int32)
        if len(faces)==0:raise ValueError(f'No triangles: {path}')
        data={'vertices':torch.as_tensor(verts,device=self.device),
              'faces':torch.as_tensor(faces,device=self.device)}
        vis=mesh.visual;uv=getattr(vis,'uv',None);material=getattr(vis,'material',None)
        tex_img=getattr(material,'image',None) if material is not None else None
        if uv is not None and tex_img is not None:
            uv=np.asarray(uv,dtype=np.float32)
            if uv.shape!=(len(verts),2):raise ValueError(f'Unsupported UV: {path} {uv.shape}')
            tex=np.asarray(tex_img.convert('RGB') if isinstance(tex_img,Image.Image)
                           else Image.fromarray(np.asarray(tex_img)).convert('RGB'),dtype=np.float32)/255.
            data.update(mode='texture',uv=torch.as_tensor(uv,device=self.device),
                        tex=torch.as_tensor(tex,device=self.device).contiguous())
        elif self.allow_vertex_colors and getattr(vis,'kind',None)=='vertex':
            rgba=np.asarray(vis.vertex_colors)
            data.update(mode='vertex',colors=torch.tensor(rgba[:,:3].astype(np.float32)/255.,device=self.device))
        else:raise ValueError(f'Mesh {path} has no usable texture or vertex colors')
        self.cache[obj_id]=data
        print(f'Loaded {path.name}: {len(verts)} vertices, {len(faces)} faces, mode={data["mode"]}',flush=True)
        return data
    def render(self,obj_id,R,t,K):
        mesh=self._load(int(obj_id));v=mesh['vertices'];B=R.shape[0]
        vc=torch.einsum('bij,vj->bvi',R.float(),v)+t.float()[:,None,:]
        X,Y,Z=vc.unbind(-1)
        fx,fy=K[:,0,0,None],K[:,1,1,None]
        cx,cy=K[:,0,2,None],K[:,1,2,None]
        W=H=self.size
        xclip=2*(fx*X+cx*Z)/W-Z
        yclip=Z-2*(fy*Y+cy*Z)/H
        zclip=((self.far+self.near)/(self.far-self.near))*Z-(2*self.far*self.near)/(self.far-self.near)
        pos=torch.stack([xclip,yclip,zclip,Z],dim=-1).contiguous()
        rast,_=dr.rasterize(self.ctx,pos,mesh['faces'],resolution=[H,W],grad_db=False)
        if mesh['mode']=='texture':
            uv,_=dr.interpolate(mesh['uv'][None].expand(B,-1,-1).contiguous(),rast,mesh['faces'])
            uv=torch.stack([uv[...,0],1-uv[...,1]],dim=-1)
            rgb=dr.texture(mesh['tex'][None].expand(B,-1,-1,-1).contiguous(),uv,
                           filter_mode='linear',boundary_mode='clamp')
        else:
            rgb,_=dr.interpolate(mesh['colors'][None].expand(B,-1,-1).contiguous(),rast,mesh['faces'])
        rgb=rgb*(rast[...,3:4]>0).to(rgb.dtype)
        rgb=dr.antialias(rgb.contiguous(),rast,pos,mesh['faces'])
        return rgb.flip(1).clamp(0.,1.).permute(0,3,1,2).contiguous()


class EnergyNet(nn.Module):
    GRID,PATCH=16,16
    def __init__(self,cfg,checkpoint,models_dir,device='cuda',amp=False,render_chunk=4,allow_vertex_colors=True):
        super().__init__()
        self.dev=torch.device(device);self.amp=amp;self.img_size=self.GRID*self.PATCH
        self.render_chunk=render_chunk
        self.encoder2d=hydra.utils.instantiate(cfg.encoder2d)
        try:ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
        except Exception:ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.encoder2d.load_state_dict(ckpt['models']['encoder2d'],strict=True)
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.to(self.dev).eval()
        for p in self.encoder2d.parameters():p.requires_grad_(False)
        self.renderer=NvRenderer(models_dir,device,self.img_size,allow_vertex_colors=allow_vertex_colors)
        self._target=self._tok=self._groups=self._inv=None
    def _autocast(self):return torch.autocast('cuda',dtype=torch.float16,enabled=self.amp)
    @torch.no_grad()
    def prepare(self,b):
        with self._autocast():self._target=self.encoder2d((b['img']-self.mean)/self.std).float()
        self._tok=F.avg_pool2d(b['mask'],self.PATCH).flatten(1)
        groups=defaultdict(list)
        for i,o in enumerate(b['obj_id'].tolist()):groups[int(o)].append(i)
        self._groups=[];order=[]
        for obj,ids in groups.items():
            self._groups.append((obj,torch.tensor(ids,device=self.dev)));order+=ids
        if len(self._groups)>1:
            inv=torch.empty(len(order),dtype=torch.long)
            inv[torch.tensor(order)]=torch.arange(len(order))
            self._inv=inv.to(self.dev)
        else:self._inv=None
    def forward(self,img,mask,depth,K,obj_id,R,t):
        assert self._target is not None,'Call prepare first'
        N=R.shape[1];Es=[]
        for obj,idx in self._groups:
            G=idx.numel();GN=G*N
            Ri=R[idx].reshape(GN,3,3);ti=t[idx].reshape(GN,3)
            Ki=K[idx].repeat_interleave(N,dim=0)
            tgt=self._target[idx][:,None];w=self._tok[idx][:,None,:]
            energies=[]
            for start in range(0,GN,self.render_chunk):
                end=min(start+self.render_chunk,GN)
                rgb=self.renderer.render(obj,Ri[start:end],ti[start:end],Ki[start:end])
                normalized=(rgb-self.mean)/self.std
                def encode(x):
                    with self._autocast():return self.encoder2d(x).float()
                if torch.is_grad_enabled():
                    features=grad_checkpoint(encode,normalized,use_reentrant=False)
                else:features=encode(normalized)
                if features.ndim!=3:raise RuntimeError(f'Unexpected output: {features.shape}')
                indices=torch.arange(start,end,device=self.dev)//N
                target=tgt[:,0][indices];weights=w[:,0][indices]
                err=(features-target).abs().mean(-1)
                energies.append((err*weights).sum(-1)/weights.sum(-1).clamp(min=1.))
            Es.append(torch.cat(energies).reshape(G,N))
        out=torch.cat(Es,dim=0)
        return out if self._inv is None else out[self._inv]


def _energy(net,b,R,t):
    return net(b['img'],b['mask'],b['depth'],b['K'],b['obj_id'],R,t)

@torch.no_grad()
def sample_and_score(net,b,grid,chunk):
    B,N,dev=b['t0'].shape[0],grid.shape[0],grid.device
    R=random_rotations((B,1),dev)@grid[None]
    t=b['t0'][:,None,:].expand(B,N,3).contiguous()
    E=torch.cat([_energy(net,b,R[:,i:i+chunk],t[:,i:i+chunk]) for i in range(0,N,chunk)],dim=1)
    return R,t,E


def refine(net,b,R,t,steps,lr_t,lr_R,tol=1e-5,patience=20,
           lr_end_frac=.02,min_frac=.5,trace=False):
    """Same optimizer/trace semantics as PoseJEPA; records current poses and gradients."""
    R=R.detach().clone();t=t.detach().clone().requires_grad_(True)
    w=torch.zeros_like(t,requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr_R},{'params':[t],'lr':lr_t}])
    def factor(k):
        c=.5*(1.+math.cos(math.pi*min(k,steps)/steps))
        return lr_end_frac+(1.-lr_end_frac)*c
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=factor)
    best_E=torch.full(R.shape[:2],float('inf'),device=R.device)
    best_R,best_t=R.clone(),t.detach().clone();stall=0
    tr_E=[];tr_R=[];tr_t=[];tr_gr=[];tr_gt=[]
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        current_R=so3_exp(w)@R
        E=_energy(net,b,current_R,t)
        E.sum().backward()
        if trace:
            if w.grad is None or t.grad is None:
                raise RuntimeError('Missing gradients: renderer -> DINO -> pose chain is broken')
            if not torch.isfinite(w.grad).all() or not torch.isfinite(t.grad).all():
                raise FloatingPointError(f'Nonfinite pose gradient at step {it}')
        with torch.no_grad():
            if trace:
                tr_E.append(E.detach().cpu().clone())
                tr_R.append(current_R.detach().cpu().clone())
                tr_t.append(t.detach().cpu().clone())
                tr_gr.append(w.grad.detach().float().norm(dim=-1).cpu().clone())
                tr_gt.append(t.grad.detach().float().norm(dim=-1).cpu().clone())
            imp=E.detach()<best_E
            gain=(best_E-E.detach())[imp].max().item() if imp.any() else 0.
            best_E=torch.where(imp,E.detach(),best_E)
            best_R[imp]=current_R.detach()[imp]
            best_t[imp]=t.detach()[imp]
            stall=0 if gain>tol else stall+1
        if (not trace) and it>=int(min_frac*steps) and stall>=patience:break
        opt.step();sched.step()
        with torch.no_grad():
            R=(so3_exp(w.detach())@R).detach()
            w.zero_()
    with torch.no_grad():
        E=_energy(net,b,R,t)
        if trace:
            tr_E.append(E.detach().cpu().clone());tr_R.append(R.detach().cpu().clone())
            tr_t.append(t.detach().cpu().clone())
            tr_gr.append(torch.full_like(E.detach().cpu(),float('nan')))
            tr_gt.append(torch.full_like(E.detach().cpu(),float('nan')))
        imp=E<best_E
        best_E=torch.where(imp,E,best_E)
        best_R[imp]=R[imp];best_t[imp]=t.detach()[imp]
    tr=None
    if trace:
        tr={'E':torch.stack(tr_E),'R':torch.stack(tr_R),'t':torch.stack(tr_t),
            'grad_R':torch.stack(tr_gr),'grad_t':torch.stack(tr_gt)}
    return best_R,best_t,best_E,tr


def save_render_comparison(net,b,j,k,trace,path):
    """Render the same tracked hypothesis at step 0 and at final step."""
    obj=int(b['obj_id'][j]);K=b['K'][j:j+1]
    with torch.no_grad():
        rgb0=net.renderer.render(obj,trace['R'][0,k][None].to(K.device),
                                 trace['t'][0,k][None].to(K.device),K)[0]
        rgb1=net.renderer.render(obj,trace['R'][-1,k][None].to(K.device),
                                 trace['t'][-1,k][None].to(K.device),K)[0]
    def rgb_image(x):
        arr=(x.detach().cpu().permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)
        return Image.fromarray(arr,'RGB')
    panels=[rgb_image(b['img'][j]),rgb_image(rgb0),rgb_image(rgb1),
            Image.fromarray((b['mask'][j,0].detach().cpu().numpy()*255).astype(np.uint8)).convert('RGB')]
    size=panels[0].size;out=Image.new('RGB',(size[0]*4,size[1]+26),'white')
    d=ImageDraw.Draw(out)
    for i,(im,label) in enumerate(zip(panels,['Target crop','Initial render','Final render','CNOS mask'])):
        out.paste(im,(i*size[0],26));d.text((i*size[0]+6,7),label,fill='black')
    Path(path).parent.mkdir(parents=True,exist_ok=True);out.save(path)


def run(net,ds,args):
    dev=args.device;grid=super_fibonacci(args.N,dev)
    dl=DataLoader(ds,batch_size=args.batch_size,num_workers=args.workers,shuffle=False,
                  pin_memory=True,persistent_workers=args.workers>0)
    n_hyp=args.trace_k if args.trace else args.topk
    out,traces=[],{}
    img_saved=0;debug_rows=[]
    for b in tqdm(dl,desc='nvdiffrast trace' if args.trace else 'nvdiffrast inference'):
        b={k:v.to(dev,non_blocking=True) for k,v in b.items()}
        net.prepare(b)
        R,t,E=sample_and_score(net,b,grid,args.chunk)
        if args.trace:
            # Same sparse selection over sorted coarse energies as original experiment.
            order=E.argsort(dim=1)
            pos=torch.linspace(0,E.shape[1]-1,n_hyp,device=E.device).round().long()
            idx=order[:,pos]
        else:idx=E.topk(args.topk,dim=1,largest=False).indices
        R=torch.gather(R,1,idx[...,None,None].expand(-1,-1,3,3))
        t=torch.gather(t,1,idx[...,None].expand(-1,-1,3))
        with torch.enable_grad():
            R,t,E,tr=refine(net,b,R,t,args.steps,args.lr_t,args.lr_R,trace=args.trace)
        best=E.argmin(dim=1);ar=torch.arange(len(best),device=dev)
        for j in range(len(best)):
            if not bool(b['valid'][j]):continue
            if args.trace:
                key=int(b['idx'][j]);record={
                    'scene_id':int(b['scene_id'][j]),'im_id':int(b['im_id'][j]),
                    'obj_id':int(b['obj_id'][j]),'det_score':float(b['det_score'][j]),
                    'E':tr['E'][:,j].clone(),'R':tr['R'][:,j].clone(),
                    't':tr['t'][:,j].clone(),
                    'mask':b['mask'][j,0].bool().cpu().clone(),
                    'box':b['box'][j].cpu().clone(),
                    'grad_R':tr['grad_R'][:,j].clone(),
                    'grad_t':tr['grad_t'][:,j].clone()}
                traces[key]=record
                if img_saved<args.save_images:
                    # best initial-energy candidate = selected hypothesis column 0
                    path=os.path.join(args.images_dir,f'scene{record["scene_id"]:06d}_im{record["im_id"]:06d}_obj{record["obj_id"]:06d}_idx{key}_hyp0.png')
                    save_render_comparison(net,b,j,0,record,path)
                    img_saved+=1
                if len(debug_rows)<args.debug_print:
                    debug_rows.append((key,record))
                continue
            k=int(best[j]);e=E[j,k]
            out.append({'scene_id':int(b['scene_id'][j]),'im_id':int(b['im_id'][j]),
                        'obj_id':int(b['obj_id'][j]),'score':float(-e),
                        'R':R[ar[j],best[j]].detach().cpu().numpy(),
                        't':t[ar[j],best[j]].detach().cpu().numpy().reshape(3,1),'time':-1})
    for key,r in debug_rows:
        h=0
        print(f'[NVTRACE] idx={key} obj={r["obj_id"]} hyp={h}: '
              f'E0={r["E"][0,h]:.7f} Efinal={r["E"][-1,h]:.7f} '
              f'Emin={r["E"][:,h].min():.7f} '
              f'mean|gR|={r["grad_R"][:-1,h].mean():.3e} '
              f'mean|gt|={r["grad_t"][:-1,h].mean():.3e}',flush=True)
    return out,traces


def select_and_save(results,targets_json,out_csv):
    with open(targets_json) as f:
        cnt={(r['scene_id'],r['im_id'],r['obj_id']):r['inst_count'] for r in json.load(f)}
    g=defaultdict(list)
    for r in results:g[(r['scene_id'],r['im_id'],r['obj_id'])].append(r)
    final=[]
    for k,hs in g.items():
        if k in cnt:
            hs.sort(key=lambda x:x['score'],reverse=True)
            final+=hs[:cnt[k]]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)),exist_ok=True)
    inout.save_bop_results(out_csv,final)


def load_cfg(config_path,config_name,overrides):
    with initialize(version_base=None,config_path=config_path):
        return compose(config_name=config_name,overrides=list(overrides))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/data/riccardop02/datasets/ycbv')
    p.add_argument('--cnos',default='/data/riccardop02/detection/gt/gt_ycbv-test.json')
    p.add_argument('--out',default='/experiments/riccardop02/Inference')
    p.add_argument('--split',default='test');p.add_argument('--targets',default=None)
    p.add_argument('--ckpt',default='/experiments/riccardop02/outputs/2026-10-26/07-25-30/checkpoints/epoch_0040.pt')
    p.add_argument('--models_dir',default=None,help='BOP folder containing obj_000001.ply etc.; default <root>/models')
    p.add_argument('--config_path',default='configs');p.add_argument('--config_name',default='config')
    p.add_argument('--overrides',nargs='*',default=[])
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--device',default='cuda');p.add_argument('--min_score',type=float,default=0.)
    p.add_argument('--det_per_target',type=int,default=0)
    p.add_argument('--crop_ratio',type=float,default=1.2)
    p.add_argument('--N',type=int,default=128);p.add_argument('--topk',type=int,default=3)
    p.add_argument('--chunk',type=int,default=32);p.add_argument('--render_chunk',type=int,default=4)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--lr_t',type=float,default=2.)
    p.add_argument('--lr_R',type=float,default=.03)
    p.add_argument('--batch_size',type=int,default=8);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--no_eval',action='store_true')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--trace',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--trace_n',type=int,default=10)
    p.add_argument('--trace_k',type=int,default=6)
    p.add_argument('--trace_out',default=None)
    p.add_argument('--save_images',type=int,default=3)
    p.add_argument('--images_dir',default=None)
    p.add_argument('--debug_print',type=int,default=3)
    args=p.parse_args()
    if args.trace_k<1 or args.trace_k>args.N:p.error('--trace_k must be in [1,N]')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    targets=args.targets or f'{args.root}/test_targets_bop19.json'
    cfg=load_cfg(args.config_path,args.config_name,args.overrides)
    models_dir=args.models_dir or os.path.join(args.root,'models')
    net=EnergyNet(cfg,args.ckpt,models_dir,device=args.device,amp=args.amp,render_chunk=args.render_chunk)
    ds=CnosDetDataset(args.root,args.cnos,targets,args.split,net.img_size,
                      margin=args.crop_ratio,min_score=args.min_score,det_per_target=args.det_per_target)
    if args.trace:
        rs=np.random.RandomState(args.seed)
        sel=np.sort(rs.choice(len(ds),size=min(args.trace_n,len(ds)),replace=False))
        ds_run=Subset(ds,sel.tolist())
    else:ds_run=ds
    trace_path=args.trace_out or os.path.join(args.out,'nvdiffrast_traces.pt')
    args.images_dir=args.images_dir or os.path.join(os.path.dirname(os.path.abspath(trace_path)),'render_comparisons')
    results,traces=run(net,ds_run,args)
    if args.trace:
        os.makedirs(os.path.dirname(os.path.abspath(trace_path)),exist_ok=True)
        torch.save(traces,trace_path)
        with open(trace_path+'.config.json','w') as f:json.dump(vars(args),f,indent=2)
        print(f'Traces saved: {len(traces)} detections -> {trace_path}',flush=True)
        print(f'Render images -> {args.images_dir}',flush=True)
        return
    output=args.out
    if not output.endswith('.csv'):
        output=os.path.join(output,datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_nvdiffrast_trace',
                            f'energy_{os.path.basename(args.root.rstrip("/"))}-{args.split}.csv')
    select_and_save(results,targets,output)
    print(f'BOP results -> {output}',flush=True)
    if not args.no_eval:
        out_dir=os.path.dirname(os.path.abspath(output))
        env=os.environ.copy()
        env.update(BOP_PATH=os.path.dirname(args.root.rstrip('/')),BOP_RESULTS_PATH=out_dir,
                   BOP_EVAL_PATH=out_dir,BOP_OUTPUT_PATH=out_dir)
        subprocess.run([sys.executable,'scripts/eval_bop19_pose.py','--renderer_type','vispy',
                        '--result_filenames',os.path.basename(output)],
                       cwd='/home/riccardop02/JEPA/bop_toolkit',env=env,check=True)

if __name__=='__main__':main()
