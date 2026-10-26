from pathlib import Path
import math
from html import escape
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent / 'workflow_revised'
OUT.mkdir(exist_ok=True)
S=6000
im=Image.new('RGB',(S,S),'white'); draw=ImageDraw.Draw(im)
pdf=canvas.Canvas(str(OUT/'Figure_workflow_revised.pdf'),pagesize=(720,720))
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="720" viewBox="0 0 720 720"><rect width="720" height="720" fill="white"/>']
ink = '#17384c'
def line(a,b,dashed=False):
    draw.line((a[0]*S,(1-a[1])*S,b[0]*S,(1-b[1])*S),fill=ink,width=4)
    pdf.setStrokeColor(ink); pdf.setLineWidth(1.1); pdf.setDash(4,3) if dashed else pdf.setDash()
    pdf.line(a[0]*720,a[1]*720,b[0]*720,b[1]*720)
    dash=' stroke-dasharray="4 3"' if dashed else ''
    svg.append(f'<path d="M {a[0]*720} {(1-a[1])*720} L {b[0]*720} {(1-b[1])*720}" stroke="{ink}" stroke-width="1.1"{dash}/>')
def txt(x,y,text,fs=11.5):
    rows=text.split('\n'); step=fs*1.25
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',round(fs*S/720))
    for i,t in enumerate(rows):
        yy=(1-y)*720+(i-(len(rows)-1)/2)*step
        draw.text((x*S,yy*S/720),t,font=font,fill='#15232c',anchor='mm')
        pdf.setFillColor('#15232c'); pdf.setFont('Helvetica',fs)
        pdf.drawCentredString(x*720,720-yy-fs*.33,t)
        svg.append(f'<text x="{x*720}" y="{yy+fs*.33}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{fs}" fill="#15232c">{escape(t)}</text>')
class Ax:
    def text(self,x,y,text,fontsize=11.5,**kwargs): txt(x,y,text,fontsize)
ax=Ax()
def box(x,y,w,h,text,color='#ffffff',fs=11.5):
    draw.rounded_rectangle((x*S,(1-y-h)*S,(x+w)*S,(1-y)*S),radius=20,fill=color,outline=ink,width=4)
    pdf.setFillColor(color); pdf.setStrokeColor(ink); pdf.setDash(); pdf.roundRect(x*720,y*720,w*720,h*720,6,fill=1,stroke=1)
    svg.append(f'<rect x="{x*720}" y="{(1-y-h)*720}" width="{w*720}" height="{h*720}" rx="6" fill="{color}" stroke="{ink}" stroke-width="1.1"/>')
    txt(x+w/2,y+h/2,text,fs)
def arrow(points, dashed=False):
    for a,b in zip(points,points[1:]): line(a,b,dashed)
    a,b=points[-2:]; angle=math.atan2(b[1]-a[1],b[0]-a[0])
    for off in [-.45,.45]: line(b,(b[0]-.009*math.cos(angle+off),b[1]-.009*math.sin(angle+off)))
def header(x,w,text): box(x,.922,w,.065,text,'#e7eff4',12)
x1,x2,x3=.015,.35,.685; w=.295
header(x1,w,'A  Data preparation')
header(x2,w,'B  Source-domain\ndevelopment')
header(x3,w,'C  Cross-crop evaluation')
ys=[.805,.675,.545,.415,.285]; h=.085
texts=['Available UAV imagery\nV1 and V2', 'Orthomosaic generation\nand band alignment',
       'Spectral bands and\nvegetation indices', 'Partial reference masks\nCrop / weed / soil',
       'Paired image–mask patches\n256 × 256; stride 128']
for y,t in zip(ys,texts): box(x1,y,w,h,t)
for a,b in zip(ys,ys[1:]): arrow([(x1+w/2,a),(x1+w/2,b+h)])
source=[(.805,.085,'Four spectral configurations\nU-Net; Barley 1-V1 → V2'),
        (.660,.100,'Selected configuration\nU-Net / DeepLabv3 /\nRandom Forest'),
        (.520,.095,'Selected U-Net\nTraining on combined\nBarley 1-V1 and V2'),
        (.380,.095,'Barley 2-V2\nSpatial validation and\ncheckpoint selection'),
        (.250,.085,'Retained barley-trained\nU-Net checkpoint')]
for y,hh,t in source: box(x2,y,w,hh,t)
for (a,_,_),(b,hh,_) in zip(source,source[1:]): arrow([(x2+w/2,a),(x2+w/2,b+hh)])
arrow([(x1+w,.3275),(.33,.3275),(.33,.8475),(x2,.8475)])
box(x3,.805,w,.085,'Direct transfer to rapeseed\nNo parameter updates')
box(x3,.675,w,.085,'Controlled target adaptation\nSpatially disjoint groups','#eef4f7')
box(x3,.540,w,.095,'21 / 42 / 63 adaptation patches\nThree paired repetitions',fs=11)
box(x3,.405,w,.095,'Full fine-tuning versus\ntraining from scratch\nMatched training settings',fs=11)
box(x3,.270,w,.095,'Validation-based selection\nand early stopping\nFinal held-out test evaluation',fs=11)
for a,b,hh in [(.675,.540,.095),(.540,.405,.095),(.405,.270,.095)]:
    arrow([(x3+w/2,a),(x3+w/2,b+hh)])
# Shared checkpoint input; direct transfer and adaptation are separate branches.
arrow([(x2+w,.2925),(.665,.2925),(.665,.8475),(x3,.8475)])
arrow([(.665,.7175),(x3,.7175)])
box(x1,.090,.630,.115,'Initial exploratory rapeseed adaptation\nRandom patch subsets; partial and full fine-tuning\nOverlapping image content possible; evaluation used for selection',
    '#f5f5f5',10.5)
arrow([(.665,.2925),(.665,.1475),(x1+.630,.1475)])
box(x3,.090,w,.115,'Illustrative spatial outputs\nDerived from predictions\nNot field-validated density\nor treatment prescriptions', '#f5f5f5',10.5)
ax.text(.5,.025,'Spatial outputs are illustrative products; experimental protocols are detailed in the Methods.',
        ha='center',fontsize=9,color=ink)
svg.append('</svg>'); (OUT/'Figure_workflow_revised.svg').write_text('\n'.join(svg),encoding='utf-8')
pdf.save()
im.resize((3000,3000),Image.Resampling.LANCZOS).save(OUT/'Figure_workflow_revised.png',dpi=(300,300))
im.save(OUT/'Figure_workflow_revised.tif',dpi=(600,600),compression='tiff_lzw')
print(OUT)
