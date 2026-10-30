from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, os, sys
S=1024
P=os.environ.get("ICON_OUT", os.getcwd()+"/")
RED=(204,68,68,255); RED2=(236,92,80,255); DARK=(20,20,22,255); WHITE=(240,240,244,255)
FONT="/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"
def tile():
    img=Image.new("RGBA",(S,S),(0,0,0,0)); m=int(S*0.05); r=int(S*0.225)
    grad=Image.new("RGBA",(S,S),(0,0,0,0)); gd=ImageDraw.Draw(grad)
    for y in range(S):
        t=y/S; c=int(34-14*t); gd.line((0,y,S,y),fill=(c,c,c+2,255))
    mask=Image.new("L",(S,S),0); ImageDraw.Draw(mask).rounded_rectangle((m,m,S-m,S-m),radius=r,fill=255)
    img.paste(grad,(0,0),mask); return img
def red_grad(w,h,pix=None):
    y,x=np.mgrid[0:h,0:w]; t=np.clip((x/w)*0.7+(y/h)*0.5,0,1)
    r=np.stack([150+90*t, 40+40*t, 40+30*t],-1)
    if pix:
        img=Image.fromarray(r.astype(np.uint8)).resize((max(1,w//pix),max(1,h//pix)),Image.BOX).resize((w,h),Image.NEAREST)
        a=np.asarray(img,np.float32); g=a.mean(-1,keepdims=True); a=0.35*a+0.65*g; a=a*0.42+np.array([26,14,16]); return Image.fromarray(np.clip(a,0,255).astype(np.uint8))
    return Image.fromarray(r.astype(np.uint8))
def A():
    """Disc split by a diagonal wipe: pixel-dull left, smooth red right."""
    img=tile(); d=ImageDraw.Draw(img); c=S//2; R=int(S*0.33)
    disc=Image.new("RGBA",(S,S),(0,0,0,0)); m=Image.new("L",(S,S),0); ImageDraw.Draw(m).ellipse((c-R,c-R,c+R,c+R),fill=255)
    before=red_grad(S,S,pix=96).convert("RGBA"); after=red_grad(S,S).convert("RGBA")
    split=Image.new("L",(S,S),0); ImageDraw.Draw(split).polygon([(c+int(R*0.55),c-R-10),(S,c-R-10),(S,c+R+10),(c-int(R*0.55),c+R+10)],fill=255)
    comb=Image.composite(after,before,split); disc.paste(comb,(0,0),m); img.alpha_composite(disc)
    d=ImageDraw.Draw(img); d.line((c+int(R*0.55),c-R,c-int(R*0.55),c+R),fill=WHITE,width=int(S*0.022))
    d.ellipse((c-R,c-R,c+R,c+R),outline=(10,10,12,255),width=int(S*0.012))
    img.save(P+"iconA.png"); return img
def B():
    """Big bold S with a wipe through it."""
    img=tile(); d=ImageDraw.Draw(img)
    font=ImageFont.truetype(FONT,int(S*0.66)); tw=font.getlength("S"); x=(S-tw)/2; y=int(S*0.10)
    lm=Image.new("L",(S,S),0); ImageDraw.Draw(lm).text((x,y),"S",font=font,fill=255)
    before=red_grad(S,S,pix=48).convert("RGBA"); after=red_grad(S,S).convert("RGBA")
    split=Image.new("L",(S,S),0); ImageDraw.Draw(split).polygon([(int(S*0.58),0),(S,0),(S,S),(int(S*0.42),S)],fill=255)
    comb=Image.composite(after,before,split); img.paste(comb,(0,0),lm)
    d=ImageDraw.Draw(img); d.line((int(S*0.58),int(S*0.12),int(S*0.42),int(S*0.90)),fill=WHITE,width=int(S*0.02))
    img.save(P+"iconB.png"); return img
def C():
    """Film frame, two cells: pixel-dull left, HDR red right, one wipe."""
    img=tile(); d=ImageDraw.Draw(img); fx0,fy0,fx1,fy1=int(S*0.14),int(S*0.26),int(S*0.86),int(S*0.74)
    d.rounded_rectangle((fx0-14,fy0-14,fx1+14,fy1+14),radius=30,fill=(10,10,12,255))
    w,h=fx1-fx0,fy1-fy0; mid=fx0+int(w*0.5)
    before=red_grad(w,h,pix=72); after=red_grad(w,h)
    frame=Image.new("RGB",(w,h)); frame.paste(before,(0,0)); frame.paste(after.crop((mid-fx0,0,w,h)),(mid-fx0,0))
    fm=Image.new("L",(w,h),0); ImageDraw.Draw(fm).rounded_rectangle((0,0,w-1,h-1),radius=18,fill=255)
    img.paste(frame,(fx0,fy0),fm); d=ImageDraw.Draw(img)
    d.line((mid,fy0,mid,fy1),fill=WHITE,width=int(S*0.018))
    # sprocket holes
    for yy in (fy0-14-int(S*0.07), fy1+14+int(S*0.02)):
        for i in range(7):
            x=fx0+i*(w//6.5)+int(S*0.02)
            d.rounded_rectangle((x,yy,x+int(S*0.05),yy+int(S*0.05)),radius=8,fill=(60,60,64,255))
    img.save(P+"iconC.png"); return img
if __name__=="__main__":
    # The Studio Toolbox icon: a film frame with the before/after wipe
    # -- dull pixel plate on the left, HDR red on the right. No text,
    # so it reads at 32 px. Writes iconC.png (1024) beside this file;
    # build_dmg.sh turns AppIcon.iconset into AppIcon.icns.
    C(); print("ok")
