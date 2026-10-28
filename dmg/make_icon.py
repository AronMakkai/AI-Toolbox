from PIL import Image, ImageDraw, ImageFont
import numpy as np, cv2
S=1024; P="/tmp/claude-0/-home-claude/fed10f93-feb6-50c9-91b8-5d8648588e21/scratchpad/"
RED=(204,68,68,255); DARK=(20,20,22,255); WHITE=(235,235,240,255)
BAYER=((0,8,2,10),(12,4,14,6),(3,11,1,9),(15,7,13,5))
FONT="/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf"
def tile():
    img=Image.new("RGBA",(S,S),(0,0,0,0)); m=int(S*0.05); r=int(S*0.225)
    grad=Image.new("RGBA",(S,S),(0,0,0,0)); gd=ImageDraw.Draw(grad)
    for y in range(S):
        t=y/S; c=int(30-12*t); gd.line((0,y,S,y),fill=(c,c,c+2,255))
    mask=Image.new("L",(S,S),0); ImageDraw.Draw(mask).rounded_rectangle((m,m,S-m,S-m),radius=r,fill=255)
    img.paste(grad,(0,0),mask); return img
# ── SDF teapot ─────────────────────────────────────────────
def sd_ellipsoid(p,c,r):
    q=(p-c)/r; k0=np.linalg.norm(q,axis=-1); k1=np.linalg.norm(q/r,axis=-1)
    return k0*(k0-1.0)/np.maximum(k1,1e-6)
def sd_sphere(p,c,r): return np.linalg.norm(p-c,axis=-1)-r
def sd_capsule(p,a,b,r):
    pa=p-a; ba=b-a; h=np.clip((pa@ba)/(ba@ba),0,1)
    return np.linalg.norm(pa-h[...,None]*ba,axis=-1)-r
def sd_round_cone(p,a,b,r1,r2):
    """iq's exact round cone: a capsule whose radius goes r1 at a to r2 at b."""
    ba=b-a; l2=ba@ba; rr=r1-r2; a2=l2-rr*rr; il2=1.0/l2
    pa=p-a; y=pa@ba; z=y-l2
    x2=np.sum((pa*l2-ba*y[...,None])**2,axis=-1); y2=y*y*l2; z2=z*z*l2
    k=np.sign(rr)*rr*rr*x2
    d_b=np.sqrt(x2+z2)*il2-r2
    d_a=np.sqrt(x2+y2)*il2-r1
    d_m=(np.sqrt(x2*a2*il2)+y*rr)*il2-r1
    return np.where(np.sign(z)*a2*z2>k, d_b, np.where(np.sign(y)*a2*y2<k, d_a, d_m))
def sd_torus_seg(p,c,R,r,ax,a0,a1):
    q=p-c
    # torus in the x-y plane (axis z), limited to an angle range
    ang=np.arctan2(q[...,1],q[...,0]); ang=np.where(ang<0,ang+2*np.pi,ang)
    ok=(ang>=a0)&(ang<=a1)
    d=np.hypot(np.hypot(q[...,0],q[...,1])-R,q[...,2])-r
    # outside the arc: distance to the end caps (spheres)
    e0=c+np.array([R*np.cos(a0),R*np.sin(a0),0]); e1=c+np.array([R*np.cos(a1),R*np.sin(a1),0])
    dc=np.minimum(sd_sphere(p,e0,r),sd_sphere(p,e1,r))
    return np.where(ok,d,dc)
def scene(p):
    d=sd_ellipsoid(p,np.array([0,0.05,0]),np.array([1.05,0.72,1.05]))            # body
    d=np.minimum(d,sd_ellipsoid(p,np.array([0,0.62,0]),np.array([0.78,0.26,0.78])))  # lid
    d=np.minimum(d,sd_sphere(p,np.array([0,0.95,0]),0.14))                       # knob
    # spout: capsules along a curve to the right
    # spout: a smooth cone along a bezier (many short capsules with a
    # smoothly shrinking radius)
    P0,P1,P2=np.array([0.72,-0.05,0]),np.array([1.35,0.05,0]),np.array([1.55,0.80,0])
    bez=lambda t:(1-t)**2*P0+2*(1-t)*t*P1+t*t*P2
    N=8
    for i in range(N):
        t0,t1=i/N,(i+1)/N
        d=np.minimum(d,sd_round_cone(p,bez(t0),bez(t1),0.27-0.15*t0,0.27-0.15*t1))
    # handle: torus arc on the left
    d=np.minimum(d,sd_torus_seg(p,np.array([-1.05,0.18,0]),0.55,0.11,2,np.radians(60),np.radians(300)))
    # base ring
    d=np.minimum(d,sd_torus_seg(p,np.array([0,-0.62,0]),0.62,0.07,2,0,2*np.pi)*0+ (np.hypot(np.hypot(p[...,0],p[...,2])-0.62,p[...,1]+0.62)-0.07))
    return d
def render(w,h):
    """Raymarch the teapot; returns scene-linear luminance (HDR) and a hit mask."""
    aspect=w/h
    yy,xx=np.mgrid[0:h,0:w]
    u=(xx/w-0.5)*2*aspect*0.62; v=-(yy/h-0.5)*2*0.62
    ro=np.array([0.0,0.55,4.2]); 
    # slight downward look
    rd=np.stack([u,v-0.10,np.full_like(u,-2.2)],axis=-1); rd/=np.linalg.norm(rd,axis=-1,keepdims=True)
    # tilt the view a bit (rotate about y so spout/handle read)
    th=np.radians(-22); R=np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
    rd=rd@R.T; ro=ro@R.T
    t=np.zeros(u.shape,np.float32); hit=np.zeros(u.shape,bool)
    for _ in range(96):
        p=ro+rd*t[...,None]; d=scene(p)
        hit|=d<0.003
        t=np.where(hit,t,t+np.maximum(d*0.9,0.002))
        if hit.all(): break
    p=ro+rd*t[...,None]
    e=0.004
    n=np.stack([scene(p+np.array([e,0,0]))-scene(p-np.array([e,0,0])),
                scene(p+np.array([0,e,0]))-scene(p-np.array([0,e,0])),
                scene(p+np.array([0,0,e]))-scene(p-np.array([0,0,e]))],axis=-1)
    n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-6)
    L1=np.array([0.65,0.7,0.55]); L1/=np.linalg.norm(L1)      # key, upper right
    L2=np.array([-0.8,0.2,-0.3]); L2/=np.linalg.norm(L2)      # rim from behind left
    V=-rd
    H1=(L1+V); H1/=np.linalg.norm(H1,axis=-1,keepdims=True)
    diff=np.clip(n@L1,0,1); spec=np.clip((n*H1).sum(-1),0,1)**90
    rim=np.clip(1-np.clip((n*V).sum(-1),0,1),0,1)**3*np.clip(n@L2,0,1)
    fres=np.clip(1-np.clip((n*V).sum(-1),0,1),0,1)**2
    lum=0.14+0.95*diff+0.30*fres+5.0*spec+1.6*rim    # HDR: specular and rim well above 1
    # floor / backdrop: soft studio gradient with a contact shadow
    bg=0.04+0.16*np.exp(-(((xx/w-0.72)**2)*4+((yy/h-0.25)**2)*5))
    sh=np.exp(-(((xx/w-0.5)**2)*30+((yy/h-0.86)**2)*400))*0.6
    out=np.where(hit,lum,np.clip(bg-sh*bg,0,None))
    return out.astype(np.float32)
def hable(x):
    A,B,C,D,E,F=0.15,0.50,0.10,0.20,0.02,0.30
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F
def tone_red(t):
    t=t[...,None]; lo=np.array([18,3,5]); mid=np.array([204,68,68]); hi=np.array([255,236,228])
    return np.where(t<0.6, lo+(mid-lo)*(t/0.6), mid+(hi-mid)*((t-0.6)/0.4)).astype(np.uint8)
def build(name, pix=22, expo=1.5, wipe=0.50):
    fx0,fy0,fx1,fy1=int(S*0.14),int(S*0.20),int(S*0.86),int(S*0.68); w,h=fx1-fx0,fy1-fy0
    lin=render(w//2,h//2); lin=cv2.resize(lin,(w,h),interpolation=cv2.INTER_CUBIC)
    over=np.clip(lin-1.0,0,None); bl=cv2.GaussianBlur(over,(0,0),14)
    tm=np.clip(hable((lin+0.5*bl)*expo)/hable(np.array(7.0)),0,1)
    after=Image.fromarray(tone_red(tm),"RGB")
    # before: flat SDR, desaturated dark red, coarse pixels
    sdr=np.clip(0.12+0.5*np.clip(lin,0,0.9)/0.9,0,1)
    small=Image.fromarray((sdr*255).astype(np.uint8),"L").resize((w//pix,h//pix),Image.BOX).resize((w,h),Image.NEAREST)
    g=np.asarray(small,np.float32)/255
    before=np.stack([30+g*110, 20+g*58, 22+g*60],axis=-1).astype(np.uint8)   # desaturated dark red
    before=Image.fromarray(before,"RGB")
    mid_x=int(S*wipe); img=tile(); d=ImageDraw.Draw(img)
    d.rounded_rectangle((fx0-10,fy0-10,fx1+10,fy1+10),radius=22,fill=DARK)
    frame=Image.new("RGB",(w,h)); frame.paste(before,(0,0)); frame.paste(after.crop((mid_x-fx0,0,w,h)),(mid_x-fx0,0))
    fm=Image.new("L",(w,h),0); ImageDraw.Draw(fm).rounded_rectangle((0,0,w-1,h-1),radius=14,fill=255)
    img.paste(frame,(fx0,fy0),fm); d=ImageDraw.Draw(img)
    # wipe line stays inside the frame
    d.line((mid_x,fy0,mid_x,fy1),fill=WHITE,width=int(S*0.012))
    hw=int(S*0.045); cy=(fy0+fy1)//2; d.rounded_rectangle((mid_x-hw,cy-hw//2,mid_x+hw,cy+hw//2),radius=hw//3,fill=WHITE)
    d.line((mid_x-hw//2,cy,mid_x+hw//2,cy),fill=DARK,width=6)
    font=ImageFont.truetype(FONT, int(S*0.062)); text="STUDIO TOOLBOX"; sp=int(S*0.012)
    tw=sum(font.getlength(ch) for ch in text)+sp*(len(text)-1); x=(S-tw)/2; ty=int(S*0.095)
    for ch in text: d.text((x,ty),ch,font=font,fill=(225,225,230,255)); x+=font.getlength(ch)+sp
    ry=ty+int(S*0.042); d.line((int(S*0.10),ry,(S-tw)/2-int(S*0.03),ry),fill=RED,width=4); d.line(((S+tw)/2+int(S*0.03),ry,int(S*0.90),ry),fill=RED,width=4)
    tx0,tx1=int(S*0.10),int(S*0.90); ty0=int(S*0.745); th=int(S*0.085)
    d.rounded_rectangle((tx0,ty0,tx1,ty0+th),radius=14,fill=(12,12,13,255))
    n=18; bw=(tx1-tx0-28)//n; kk=int(n*wipe)
    for k in range(n):
        x=tx0+14+k*bw; d.rectangle((x+3,ty0+12,x+bw-3,ty0+th-12),fill=(74,10,10,255) if k<kk else (46,46,50,255))
    x=tx0+14+kk*bw; d.rectangle((x+3,ty0+8,x+bw-3,ty0+th-8),fill=RED)
    for x in (tx0+6, tx1-6-int(S*0.018)): d.rounded_rectangle((x,ty0+4,x+int(S*0.018),ty0+th-4),radius=5,fill=RED)
    py=ty0+th+int(S*0.028); c=8; ncol=(tx1-tx0)//c; lead=wipe*ncol; edge=5
    for r_ in range(3):
        for i in range(ncol):
            lit = i<lead-edge or (i<lead+edge and (lead-i)/(2*edge)+0.5 > BAYER[r_%4][i%4]/16)
            d.rectangle((tx0+i*c,py+r_*c,tx0+i*c+c-2,py+r_*c+c-2),fill=RED if lit else (26,26,28,255))
    img.save(P+f"icon{name}.png"); return img
if __name__=="__main__":
    G1=build("G1")
    sheet=Image.new("RGBA",(760,560),(40,40,44,255)); big=G1.resize((512,512),Image.LANCZOS); sm=G1.resize((128,128),Image.LANCZOS)
    sheet.paste(big,(24,24),big); sheet.paste(sm,(580,216),sm); sheet.save(P+"icon_sheet8.png"); print("ok")
