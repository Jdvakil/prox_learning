#!/usr/bin/env python3
"""Code-aligned PACT-readout diagrams: native PPTX shapes and standalone SVG.

All picture thumbnails are existing source images; boxes, text, and connectors
remain separate editable objects. No image generation is used.
"""
import base64
import html
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET
from PIL import ImageFont

from package_whole_body_figure import P, A, R, REL, CT, child, xml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'images/system_diagram'
W, H = 1840, 760
INK, GRAY = '243041', '647080'
BLUE, TEAL, PURPLE, AMBER = '2869AE', '087F79', '6A5192', 'A86D22'


class Diagram:
    def __init__(self, name):
        self.name, self.items = name, []

    def text(self, text, x, y, w, size=26, color=INK, bold=False, align='center'):
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf', round(size))
        size = min(size, size*w/max(float(font.getlength(text)), 1))
        self.items.append(dict(kind='text', text=text, x=x, y=y, w=w, h=size*1.4,
                               size=size, color=color, bold=bold, align=align))

    def box(self, title, subtitle, x, y, w, h, color=GRAY, fill='F7F9FB', size=26):
        self.items.append(dict(kind='box', x=x, y=y, w=w, h=h, color=color, fill=fill))
        lines = [title] if not subtitle else [title, subtitle]
        if subtitle:
            self.text(title, x+8, y+h/2-34, w-16, size=size, color=color, bold=True)
            self.text(subtitle, x+8, y+h/2+5, w-16, size=20, color=GRAY)
        else:
            self.text(title, x+8, y+h/2-size*.7, w-16, size=size, color=color, bold=True)

    def line(self, points, color=GRAY, dashed=False, arrow=True):
        self.items.append(dict(kind='line', points=points, color=color, dashed=dashed, arrow=arrow))

    def picture(self, path, x, y, w, h):
        self.items.append(dict(kind='picture', path=str(path), x=x, y=y, w=w, h=h))

    def svg(self, omit_text=False):
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 '<defs>'+''.join(f'<marker id="arrow_{c}" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#{c}"/></marker>' for c in [GRAY,BLUE,TEAL,PURPLE,AMBER])+'</defs>']
        for item in self.items:
            t = item['kind']
            if t == 'box':
                parts.append(f'<rect x="{item["x"]}" y="{item["y"]}" width="{item["w"]}" height="{item["h"]}" rx="10" fill="#{item["fill"]}" stroke="#{item["color"]}" stroke-width="2"/>')
            elif t == 'text' and not omit_text:
                anchor = 'middle' if item['align'] == 'center' else 'start'
                xx = item['x']+item['w']/2 if anchor == 'middle' else item['x']
                parts.append(f'<text x="{xx}" y="{item["y"]+item["size"]}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{item["size"]}" font-weight="{700 if item["bold"] else 400}" fill="#{item["color"]}" text-anchor="{anchor}">{html.escape(item["text"])}</text>')
            elif t == 'line':
                pp = ' '.join(f'{x},{y}' for x,y in item['points'])
                dash = ' stroke-dasharray="8 6"' if item['dashed'] else ''
                end = f' marker-end="url(#arrow_{item["color"]})"' if item['arrow'] else ''
                parts.append(f'<polyline points="{pp}" fill="none" stroke="#{item["color"]}" stroke-width="2.5" stroke-linejoin="round"{dash}{end}/>')
            elif t == 'picture':
                data = base64.b64encode(Path(item['path']).read_bytes()).decode()
                parts.append(f'<image x="{item["x"]}" y="{item["y"]}" width="{item["w"]}" height="{item["h"]}" href="data:image/png;base64,{data}"/>')
        return '\n'.join(parts+['</svg>'])


def overview():
    d = Diagram('system_overview')
    d.text('RGB observations', 35, 48, 260, bold=True, color=BLUE)
    d.picture(ROOT/'images/whole_body/kitchen_items/scene_with_robot.png', 35, 105, 260, 147)
    d.text('Global visual context', 30, 273, 270, size=21, color=GRAY)
    d.box('Visual encoder', '', 350, 130, 255, 100, BLUE, 'EFF5FC')
    d.box('Image tokens', '', 665, 130, 220, 100, BLUE, 'EFF5FC')
    d.line([(295,180),(350,180)], BLUE); d.line([(605,180),(665,180)], BLUE)

    d.text('Whole-body proximity', 20, 355, 290, bold=True, color=TEAL)
    for i, name in enumerate(['link6_sensor_3','link5_back_sensor_4','link2_sensor_3']):
        d.picture(ROOT/f'images/whole_body/kitchen_items/{name}/readout_8x8.png', 36+i*90, 420, 78, 78)
    d.text('40 sensors × 8 × 8 ranges', 20, 523, 290, size=20, color=GRAY)
    d.box('Temporal encoder', 'Shared across sensors', 350, 400, 255, 112, TEAL, 'EAF6F3')
    d.box('Sensor tokens', 'One per sensor', 665, 400, 220, 112, TEAL, 'EAF6F3')
    d.line([(295,456),(350,456)], TEAL); d.line([(605,456),(665,456)], TEAL)
    d.box('Joint state', 'Arm + gripper', 665, 612, 220, 90)

    d.box('PACT', 'Multimodal transformer', 1010, 205, 290, 307, PURPLE, 'F4F0F8', size=38)
    d.line([(885,180),(947,180),(947,280),(1010,280)], BLUE)
    d.line([(885,456),(947,456),(947,412),(1010,412)], TEAL)
    d.line([(885,657),(975,657),(975,484),(1010,484)])
    d.box('Action chunk', 'Arm + gripper commands', 1400, 299, 370, 118, AMBER, 'FCF6EB', size=30)
    d.line([(1300,358),(1400,358)], PURPLE)
    d.text('Execute, then query again', 1375, 455, 420, size=22, color=GRAY)
    d.text('Live RGB and proximity at each policy query', 32, 681, 570, size=21, color=GRAY, align='left')
    return d


def architecture():
    d = Diagram('readout_architecture')
    d.box('RGB camera(s)', '240 × 320 per view', 30, 90, 230, 105, BLUE, 'EFF5FC')
    d.box('ResNet-18', 'Shared visual backbone', 310, 90, 230, 105, BLUE, 'EFF5FC')
    d.box('Image tokens', '512-d + image positions', 590, 90, 260, 105, BLUE, 'EFF5FC')
    d.line([(260,143),(310,143)], BLUE); d.line([(540,143),(590,143)], BLUE)
    d.line([(850,143),(1020,143),(1020,230),(1080,230)], BLUE)

    d.box('Range history', '8 × 40 × 8 × 8', 30, 315, 230, 132, TEAL, 'EAF6F3')
    d.text('Closeness: 5–200 mm', 25, 465, 240, size=18, color=GRAY)
    d.box('Shared encoder', 'CNN + temporal transformer', 310, 290, 255, 180, TEAL, 'EAF6F3', size=25)
    d.text('Repeat each of 8 frames ×4', 310, 485, 255, size=18, color=GRAY)
    d.box('40 readouts', '128-d CLS per sensor', 615, 320, 230, 125, TEAL, 'EAF6F3')
    d.box('Projection', '512-d + sensor identity', 890, 315, 250, 132, TEAL, 'EAF6F3', size=24)
    # The fusion encoder sits to the right of the modality projections.
    d.line([(260,381),(310,381)], TEAL); d.line([(565,381),(615,381)], TEAL)
    d.line([(845,381),(890,381)], TEAL)
    d.box('ACT encoder', '4 layers · 512-d', 1190, 185, 225, 288, PURPLE, 'F4F0F8')
    # Route visual tokens above the sensor projection, then into fusion.
    d.items = [i for i in d.items if not (i['kind']=='line' and i['points']==[(850,143),(1020,143),(1020,230),(1080,230)])]
    d.line([(850,143),(1148,143),(1148,240),(1190,240)], BLUE)
    d.line([(1140,381),(1190,381)], TEAL)
    d.box('ACT decoder', '7 layers', 1480, 220, 220, 210, PURPLE, 'F4F0F8')
    d.line([(1415,330),(1480,330)], PURPLE)
    d.box('50 queries', 'Learned action positions', 1465, 64, 250, 102, PURPLE, 'F4F0F8', size=25)
    d.line([(1590,166),(1590,220)], PURPLE)
    d.box('Action chunk', '50 × 8 commands', 1470, 505, 245, 102, AMBER, 'FCF6EB')
    d.line([(1590,430),(1590,505)], PURPLE)
    d.box('Joint state', '9-d → 512-d token', 890, 520, 250, 100)
    d.line([(1140,570),(1170,570),(1170,438),(1190,438)])
    d.box('z = 0', 'Latent token at inference', 1190, 530, 225, 100, size=25)
    d.line([(1303,530),(1303,473)])

    d.box('Geometry pretraining', 'XYZ · validity · depth · future', 300, 588, 485, 94, TEAL, 'F7FAF9', size=24)
    d.line([(300,635),(285,635),(285,425),(310,425)], TEAL, dashed=True)
    d.text('Initialize shared encoder', 300, 537, 320, size=18, color=TEAL)
    d.text('Inference path shown. History sampling and camera selection depend on the run.', 30, 718, 1550, size=19, color=GRAY, align='left')
    return d


def xform(parent, x, y, w, h, flip_h=False, flip_v=False):
    xf = child(parent, A, 'xfrm')
    if flip_h: xf.set('flipH', '1')
    if flip_v: xf.set('flipV', '1')
    child(xf, A, 'off', x=round(x*6000), y=round(y*6000))
    child(xf, A, 'ext', cx=round(w*6000), cy=round(h*6000))


def solid(parent, color):
    child(child(parent, A, 'solidFill'), A, 'srgbClr', val=color)


def pptx(diagrams):
    template = ROOT/'images/whole_body/whole_body_canva.pptx'
    with ZipFile(template) as z:
        files = {n:z.read(n) for n in z.namelist() if not n.startswith(('ppt/slides/','ppt/media/')) and n!='docProps/thumbnail.jpeg'}
    pres = ET.fromstring(files['ppt/presentation.xml']); ids = pres.find(f'{{{P}}}sldIdLst'); ids.clear()
    size = pres.find(f'{{{P}}}sldSz'); size.set('cx', str(W*6000)); size.set('cy', str(H*6000))
    rels = ET.fromstring(files['ppt/_rels/presentation.xml.rels'])
    for rel in list(rels):
        if rel.get('Type','').endswith('/slide'): rels.remove(rel)
    types = ET.fromstring(files['[Content_Types].xml'])
    for item in list(types):
        if item.get('PartName','').startswith('/ppt/slides/'): types.remove(item)
    rootrels = ET.fromstring(files['_rels/.rels'])
    for rel in list(rootrels):
        if rel.get('Type','').endswith('/thumbnail'): rootrels.remove(rel)
    files['_rels/.rels'] = xml(rootrels)
    for page, diagram in enumerate(diagrams, 1):
        slide = ET.Element(f'{{{P}}}sld'); cs = child(slide, P, 'cSld', name=diagram.name)
        tree = child(cs, P, 'spTree'); nv = child(tree, P, 'nvGrpSpPr')
        child(nv, P, 'cNvPr', id=1, name=''); child(nv, P, 'cNvGrpSpPr'); child(nv, P, 'nvPr'); child(tree, P, 'grpSpPr')
        sr = ET.Element(f'{{{REL}}}Relationships')
        child(sr, REL, 'Relationship', Id='rId1', Type=R+'/slideLayout', Target='../slideLayouts/slideLayout7.xml')
        shape_id = 2
        for item in diagram.items:
            kind = item['kind']
            parts = [(item['points'][j], item['points'][j+1], j==len(item['points'])-2) for j in range(len(item['points'])-1)] if kind=='line' else [None]
            for segment in parts:
                if kind=='picture':
                    element = child(tree, P, 'pic'); nv = child(element, P, 'nvPicPr')
                    child(nv, P, 'cNvPr', id=shape_id, name=Path(item['path']).name)
                    child(nv, P, 'cNvPicPr'); child(nv, P, 'nvPr')
                    media = f'diagram_{page}_{shape_id}.png'; files['ppt/media/'+media] = Path(item['path']).read_bytes()
                    child(sr, REL, 'Relationship', Id=f'rId{shape_id}', Type=R+'/image', Target='../media/'+media)
                    fill = child(element, P, 'blipFill'); b = child(fill, A, 'blip'); b.set(f'{{{R}}}embed', f'rId{shape_id}')
                    child(child(fill, A, 'stretch'), A, 'fillRect')
                    sp = child(element, P, 'spPr'); xform(sp,item['x'],item['y'],item['w'],item['h'])
                    child(child(sp,A,'prstGeom',prst='rect'),A,'avLst')
                elif kind=='line':
                    (x1,y1),(x2,y2),last = segment
                    element = child(tree,P,'cxnSp'); nv = child(element,P,'nvCxnSpPr')
                    child(nv,P,'cNvPr',id=shape_id,name='Flow connector');child(nv,P,'cNvCxnSpPr');child(nv,P,'nvPr')
                    sp=child(element,P,'spPr');xform(sp,min(x1,x2),min(y1,y2),abs(x2-x1),abs(y2-y1),x2<x1,y2<y1)
                    child(child(sp,A,'prstGeom',prst='line'),A,'avLst')
                    line=child(sp,A,'ln',w=15000);solid(line,item['color'])
                    if item['dashed']:child(line,A,'prstDash',val='dash')
                    if last and item['arrow']:child(line,A,'tailEnd',type='triangle',w='med',len='med')
                else:
                    element=child(tree,P,'sp');nv=child(element,P,'nvSpPr')
                    child(nv,P,'cNvPr',id=shape_id,name=item.get('text','Module box'))
                    child(nv,P,'cNvSpPr',**({'txBox':'1'} if kind=='text' else {}));child(nv,P,'nvPr')
                    sp=child(element,P,'spPr');xform(sp,item['x'],item['y'],item['w'],item['h'])
                    child(child(sp,A,'prstGeom',prst='roundRect' if kind=='box' else 'rect'),A,'avLst')
                    if kind=='box':
                        solid(sp,item['fill']);line=child(sp,A,'ln',w=12000);solid(line,item['color'])
                    else:
                        child(sp,A,'noFill');child(child(sp,A,'ln'),A,'noFill')
                        body=child(element,P,'txBody');child(body,A,'bodyPr',wrap='none',lIns=0,rIns=0,tIns=0,bIns=0,anchor='t');child(body,A,'lstStyle')
                        para=child(body,A,'p');child(para,A,'pPr',algn='ctr' if item['align']=='center' else 'l')
                        run=child(para,A,'r');rp=child(run,A,'rPr',lang='en-US',sz=round(item['size']*6000/127),b='1' if item['bold'] else '0')
                        solid(rp,item['color']);child(rp,A,'latin',typeface='DejaVu Sans');child(run,A,'t').text=item['text']
                shape_id += 1
        child(child(slide,P,'clrMapOvr'),A,'masterClrMapping')
        files[f'ppt/slides/slide{page}.xml']=xml(slide);files[f'ppt/slides/_rels/slide{page}.xml.rels']=xml(sr)
        node=child(ids,P,'sldId',id=255+page);node.set(f'{{{R}}}id',f'rId{50+page}')
        child(rels,REL,'Relationship',Id=f'rId{50+page}',Type=R+'/slide',Target=f'slides/slide{page}.xml')
        child(types,CT,'Override',PartName=f'/ppt/slides/slide{page}.xml',ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
    files['ppt/presentation.xml']=xml(pres);files['ppt/_rels/presentation.xml.rels']=xml(rels);files['[Content_Types].xml']=xml(types)
    core=ET.fromstring(files['docProps/core.xml'])
    for node in core:
        if node.tag.endswith('}title'):node.text='PACT system diagram'
        if node.tag.endswith('}description'):node.text='Code-aligned overview and readout architecture. Native editable shapes, arrows, and labels.'
    files['docProps/core.xml']=xml(core)
    with ZipFile(OUT/'system_diagram_canva.pptx','w',ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    diagrams=[overview(),architecture()]
    blank=Diagram('system_overview_unlabelled');blank.items=[i for i in diagrams[0].items if i['kind']!='text']
    for d in diagrams+[blank]:
        (OUT/f'{d.name}.svg').write_text(d.svg())
    pptx(diagrams+[blank])
    manifest=dict(method='PACT-readout inference path',
        sources=['README.md §10','encoders/surface_geometry.py:390–463,875–914',
                 'submodules/act/detr/models/detr_vae.py:109–127,185–240',
                 'submodules/act/detr/models/transformer.py:53–91','eval_act.py:707–838,1158'],
        camera_note='Overview thumbnail is the existing simulator re-render at the same lowered pose as the range thumbnails, using original dataset geometry. Camera count/selection varies by run; hallway headline uses wrist RGB.',
        range_thumbnails='Three previously exported simulator range panels illustrate different body locations, not policy attention or intermediate features.',
        range_preprocessing='Per-sensor valid depths 0.005–0.20 m mapped to closeness; out-of-range/dead pixels become zero. Overview thumbnail colours use the figure scale, not this policy preprocessing.',
        temporal_note='8 pooled observations are repeated four times for the 32-frame shared temporal encoder. Training uses consecutive steps; eval_act.py defaults to query-spaced history. Diagram does not imply continuous reflex control.',
        architecture='40 shared-encoder CLS readouts of width128 projected to 40 tokens of width512; sensor identity is supplied by learned ACT positional embeddings. ACT memory also includes image, qpos and latent tokens.',
        pretraining='Auxiliary geometry targets supervise encoder pretraining. These heads are not runtime inputs; main readout is not a reconstructed object map. Policy training finetunes the encoder.',
        inference='z=0; action chunk of50 commands with8 dimensions for the documented readout configuration. Query cadence and camera choice depend on run.',
        editing='PPTX has native boxes, separate editable text and arrow segments; source thumbnails are PNGs. SVGs embed images and retain vector text/shapes. Third slide has no labels.',
        canvas=[W,H])
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'caption.txt').write_text('PACT combines global visual context with local range measurements distributed over the arm. A shared temporal encoder produces one feature per proximity sensor. Image, sensor, and joint-state tokens condition an ACT transformer that predicts arm and gripper action chunks. Fresh observations are consumed at policy queries. Geometry supervision is used to pretrain the sensor encoder; no explicit object reconstruction is required at inference.\n')
    (OUT/'diagram_layout.json').write_text(json.dumps({d.name:d.items for d in diagrams},indent=2))
    print('Saved editable system diagrams to',OUT)


if __name__=='__main__':main()
