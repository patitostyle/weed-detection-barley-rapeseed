"""Compact supporting tables covering all 224 neural training runs."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'checkpoint_epochs'


def main():
    data=json.loads((OUT/'checkpoint_epochs.json').read_text())
    rows=data['runs']
    covered=set()
    styles=getSampleStyleSheet()
    for s in styles.byName.values():s.textColor=colors.black
    styles.add(ParagraphStyle(name='DocTitle',fontName='Helvetica-Bold',fontSize=17,leading=21,spaceAfter=10))
    styles.add(ParagraphStyle(name='TextSmall',fontName='Helvetica',fontSize=10,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle(name='TableTitle',fontName='Helvetica-Bold',fontSize=11,leading=14,spaceBefore=9,spaceAfter=5))
    styles.add(ParagraphStyle(name='CellText',fontName='Helvetica',fontSize=9.5,leading=12))
    styles.add(ParagraphStyle(name='CellHead',fontName='Helvetica-Bold',fontSize=9.5,leading=12,alignment=TA_CENTER))
    def p(s,style='TextSmall'):return Paragraph(s,styles[style])
    def table(values,widths,left_cols=()):
        content=[]
        for i,row in enumerate(values):
            content.append([p(str(x),'CellHead' if i==0 else 'CellText') for x in row])
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        cmds=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EDF1')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
              ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('LEFTPADDING',(0,0),(-1,-1),5),
              ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]
        t.setStyle(TableStyle(cmds))
        # Paragraph alignment must be set explicitly; Table ALIGN does not
        # change the alignment of text inside Paragraph objects.
        for i,row in enumerate(t._cellvalues):
            for j,cell in enumerate(row):
                cell.style=ParagraphStyle(name=f'align{i}_{j}',parent=cell.style,
                                          alignment=TA_LEFT if i>0 and j in left_cols else TA_CENTER)
        return t
    def find(**kw):
        r=[r for r in rows if all(r[k]==v for k,v in kw.items())]
        assert len(r)==1,(kw,len(r))
        covered.add(r[0]['run_path'])
        return r[0]
    story=[p('Selected checkpoint epochs','DocTitle'),
           p('Supporting tables for Reviewer 1 Comment 4.7'),
           p('The tables report selected epochs for all 205 target-domain runs and 19 distinct source-domain neural runs used in the revised study. Epochs are numbered from one. Random Forest fits have no epoch-based checkpoint selection. Direct transfer involves no target training.'),
           p('Checkpoint selection used validation weed F1 only, with a required improvement greater than 0.0001. Training was capped at 30 epochs and stopped after seven consecutive epochs without improvement. The CSV record also reports the actual stopping epoch and validation score for every run.'),
           p('Table 1 Primary rapeseed comparisons','TableTitle'),
           p('Each cell gives scratch / full fine-tuning selected epochs. A complete adaptation pool contains 224-336 patches, depending on the held-out group. Normalization was patch-wise.')]
    body=[['Test group','Adaptation<br/>patches']+[f'Seed {s}' for s in [42,43,44,45,46]]]
    for g in [2,4,6,7,8]:
        for budget in ['21','63','complete pool']:
            rr=[r for r in rows if r['domain']=='target' and r['test_group']==g and r['budget']==budget and r['stage'] in [1,2]]
            actual=rr[0]['adaptation_patches']
            line=[str(g),f'{actual} (all)' if budget=='complete pool' else budget]
            for seed in [42,43,44,45,46]:
                x=find(domain='target',test_group=g,budget=budget,seed=seed,strategy='scratch',stage=1 if budget=='63' else 2)
                y=find(domain='target',test_group=g,budget=budget,seed=seed,strategy='full',stage=1 if budget=='63' else 2)
                line.append(f"{x['selected_epoch']} / {y['selected_epoch']}")
            body.append(line)
    story+=[table(body,[52,88]+[70]*5),Spacer(1,8),
            p('These 150 runs constitute the primary scratch versus full fine-tuning comparisons. The group identifiers match the spatial-fold definitions in the manuscript.'),PageBreak(),
            p('Complementary experiments and source models','DocTitle'),
            p('Table 2 Partial fine tuning at 63 patches','TableTitle'),
            p('Selected epoch for each of the 25 runs using patch-wise normalization.')]
    body=[['Test group']+[f'Seed {s}' for s in [42,43,44,45,46]]]
    for g in [2,4,6,7,8]:
        body.append([g]+[find(domain='target',stage=4,test_group=g,seed=s,strategy='partial')['selected_epoch'] for s in [42,43,44,45,46]])
    story+=[table(body,[70]+[84]*5),p('Table 3 Source fixed normalization at 63 patches','TableTitle'),
            p('Each cell gives scratch / full fine-tuning selected epochs. Only seeds 42-44 were run in this complementary comparison.')]
    body=[['Test group']+[f'Seed {s}' for s in [42,43,44]]]
    for g in [2,4,6,7,8]:
        line=[g]
        for seed in [42,43,44]:
            x=find(domain='target',stage=3,test_group=g,seed=seed,strategy='scratch')
            y=find(domain='target',stage=3,test_group=g,seed=seed,strategy='full')
            line.append(f"{x['selected_epoch']} / {y['selected_epoch']}")
        body.append(line)
    story+=[table(body,[70,140,140,140]),p('Table 4 Source domain neural models','TableTitle'),
            p('Patch-wise normalization applies unless source-fixed scaling is specified. A dash means that no run was scheduled. Reused checkpoints are counted once.')]
    body=[['Model and inputs']+[f'Seed {s}' for s in [42,43,44]]]
    specs=[('U-Net RGB','U-Net','RGB','patch_valid'),
           ('U-Net RGB plus NIR and RE','U-Net','RGB + NIR + RE','patch_valid'),
           ('U-Net RGB plus indices','U-Net','RGB + indices','patch_valid'),
           ('U-Net eight channels','U-Net','eight channels','patch_valid'),
           ('DeepLabv3 random','DeepLabv3 random','eight channels','patch_valid'),
           ('DeepLabv3 ImageNet backbone','DeepLabv3 imagenet','eight channels','patch_valid'),
           ('U-Net eight channels source-fixed','U-Net','eight channels','source_fixed')]
    for label,model,inputs,norm in specs:
        line=[label]
        for seed in [42,43,44]:
            if norm=='source_fixed' and seed!=42:line.append('-')
            else:line.append(find(domain='source',model=model,input=inputs,normalization=norm,seed=seed)['selected_epoch'])
        body.append(line)
    story.append(table(body,[265,75,75,75],left_cols=(0,)))
    assert len(covered)==224 and covered=={r['run_path'] for r in rows}
    def footer(c,doc):
        c.saveState();c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#555555'))
        c.drawString(48,23,'Geomatics geomatics-4689896 | Checkpoint selection records')
        c.drawRightString(letter[0]-48,23,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(OUT/'checkpoint_epochs_reviewer1.pdf'),pagesize=letter,
                          rightMargin=48,leftMargin=48,topMargin=36,bottomMargin=38,
                          title='Selected checkpoint epochs for Reviewer 1',author='Patricio Alonso Hernandez')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (OUT/'report_coverage_check.json').write_text(json.dumps({'distinct_runs_shown':len(covered),'target_runs':205,'source_runs':19,'all_runs_included':True},indent=2)+'\n')
    print('PDF exported; all224 runs represented exactly once.')


if __name__=='__main__':main()
