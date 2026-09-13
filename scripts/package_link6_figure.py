#!/usr/bin/env python3
"""Package sensor-only reconstruction plates for Canva without dense shape loops."""
from pathlib import Path
import json
import zipfile
from PIL import Image
from pptx import Presentation
from pptx.util import Inches
from pptx.oxml import parse_xml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'images/link6_reconstruction'
PALETTE=['087F8C','3964AD','7762A3','B0753A','348775','407DBA']


def main():
    prs=Presentation()
    prs.slide_width,prs.slide_height=Inches(7),Inches(5.2)
    for name in ['link6_hood_scan','link6_snapshot_384','link6_obstacle_detail']:
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        if name!='link6_snapshot_384':
            # Dense scans remain replaceable transparent images for responsive
            # Canva editing; the companion SVG keeps every measured point vector.
            im=Image.open(OUT/f'{name}.png');w,h=im.size
            scale=min(6.8/w,5/h);ww,hh=w*scale,h*scale
            pic=slide.shapes.add_picture(str(OUT/f'{name}.png'),Inches((7-ww)/2),
                 Inches((5.2-hh)/2),width=Inches(ww),height=Inches(hh))
            pic.name='Measured points only — transparent image; vector SVG supplied separately'
        else:
            layout=json.loads((OUT/f'{name}_layout.json').read_text())
            lo,hi=layout['bounds'];scale=min(6.7/(hi[0]-lo[0]),4.9/(hi[1]-lo[1]))
            offset=((7-(hi[0]-lo[0])*scale)/2,(5.2-(hi[1]-lo[1])*scale)/2)
            groups=[]
            for i in range(6):
                g=slide.shapes.add_group_shape();g.name=f'link6_sensor_{i}: 64 raw 8×8 returns';groups.append(g)
            for idx in layout['order']:
                p=layout['points_xy'][idx];sensor=layout['sensor_id'][idx]
                x=offset[0]+(p[0]-lo[0])*scale;y=offset[1]+(hi[1]-p[1])*scale;r=.018
                # Append shape XML in bulk; compute group bounds once below.
                xml=f'''<p:sp xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:nvSpPr><p:cNvPr id="{100+idx}" name="sensor {sensor}, sample {idx%64}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{int(Inches(x-r))}" y="{int(Inches(y-r))}"/><a:ext cx="{int(Inches(2*r))}" cy="{int(Inches(2*r))}"/></a:xfrm><a:prstGeom prst="ellipse"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{PALETTE[sensor]}"/></a:solidFill><a:ln><a:noFill/></a:ln><a:effectLst/></p:spPr></p:sp>'''
                groups[sensor]._element.append(parse_xml(xml))
            for g in groups:
                g._element.recalculate_extents()
                assert len(g.shapes)==64
        slide.notes_slide.notes_text_frame.text=(OUT/'caption.txt').read_text()+f'\nPlate: {name}. Snapshot: 384 individually editable points in six sensor groups. Dense scans: transparent PNG, with SVG and full 3D PLY alongside. See manifest.json for provenance and limitations.'
    prs.core_properties.title='Link 6: reconstruction from recorded 8×8 proximity measurements'
    prs.save(OUT/'link6_canva.pptx')
    with zipfile.ZipFile(OUT/'link6_canva_assets.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.suffix in ['.png','.svg','.pdf','.pptx','.txt','.csv'] or p.name=='manifest.json':
                z.write(p,p.name)
    print('Packaged three sensor-only slides; snapshot contains six editable groups of 64 points.')


if __name__=='__main__':main()
