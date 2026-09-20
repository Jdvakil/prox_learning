#!/usr/bin/env python3
"""Assemble text-free, individually editable image panels for Canva/PPTX."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import json
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'images/whole_body'
P = 'http://schemas.openxmlformats.org/presentationml/2006/main'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
for prefix, ns in [('p', P), ('a', A), ('r', R)]: ET.register_namespace(prefix, ns)


def child(parent, ns, tag, **attrs):
    return ET.SubElement(parent, f'{{{ns}}}{tag}', {k: str(v) for k, v in attrs.items()})


def xml(element):
    return ET.tostring(element, encoding='utf-8', xml_declaration=True)


def main():
    source = ROOT / 'images/link6_reconstruction/link6_canva.pptx'
    with ZipFile(source) as z:
        files = {n: z.read(n) for n in z.namelist()
                 if not n.startswith(('ppt/slides/', 'ppt/notesSlides/', 'ppt/notesMasters/', 'ppt/media/'))}
    presentation = ET.fromstring(files['ppt/presentation.xml'])
    slides = presentation.find(f'{{{P}}}sldIdLst'); slides.clear()
    notes = presentation.find(f'{{{P}}}notesMasterIdLst')
    if notes is not None: presentation.remove(notes)
    size = presentation.find(f'{{{P}}}sldSz')
    # 2496x1120 design, with each PNG its own freely movable picture element.
    scale = 5000
    size.set('cx', str(2496*scale)); size.set('cy', str(1120*scale)); size.set('type', 'custom')
    rels = ET.fromstring(files['ppt/_rels/presentation.xml.rels'])
    for rel in list(rels):
        if rel.get('Type', '').endswith(('/slide', '/notesMaster')): rels.remove(rel)
    content_types = ET.fromstring(files['[Content_Types].xml'])
    for item in list(content_types):
        if item.get('PartName', '').startswith(('/ppt/slides/', '/ppt/notesSlides/', '/ppt/notesMasters/')):
            content_types.remove(item)
    layouts = []
    for slide_index, label in enumerate(['kitchen_items', 'cabinet_cavity', 'fumehood_clutter'], 1):
        folder = OUT / label
        manifest = json.loads((folder/'manifest.json').read_text())
        with Image.open(folder/'scene_with_robot.png') as im:
            # Crop only empty side margins; preserve the vertical extent of the arm.
            crop = (380, 0, 2115, 1408) if label != 'cabinet_cavity' else (470, 0, 2205, 1408)
            im.crop(crop).save(folder/'scene_focus.png')
        panels = [(folder/'scene_focus.png', 0, 0, 1380, 1120)]
        for row, sensor in enumerate(manifest['selected_sensors']):
            for column, filename in enumerate(['rgb.png', 'depth.png', 'readout_8x8.png']):
                panels.append((folder/sensor/filename, 1416+column*360, 23+row*365, 350, 350))
        preview = Image.new('RGB', (2496, 1120), 'white')
        for path, x, y, w, h in panels:
            with Image.open(path) as im:
                method = Image.Resampling.NEAREST if path.name == 'readout_8x8.png' else Image.Resampling.LANCZOS
                preview.paste(im.resize((w, h), method), (x, y))
        preview.save(OUT/f'first_page_{label}.png')
        sld = ET.Element(f'{{{P}}}sld')
        cs = child(sld, P, 'cSld'); tree = child(cs, P, 'spTree')
        nv = child(tree, P, 'nvGrpSpPr'); child(nv, P, 'cNvPr', id=1, name='')
        child(nv, P, 'cNvGrpSpPr'); child(nv, P, 'nvPr'); child(tree, P, 'grpSpPr')
        slide_rels = ET.Element(f'{{{REL}}}Relationships')
        child(slide_rels, REL, 'Relationship', Id='rId1', Type=R+'/slideLayout', Target='../slideLayouts/slideLayout7.xml')
        for idx, (path, x, y, w, h) in enumerate(panels, 2):
            media = f'whole_body_{slide_index}_{idx}.png'
            files['ppt/media/'+media] = path.read_bytes()
            child(slide_rels, REL, 'Relationship', Id=f'rId{idx}', Type=R+'/image', Target='../media/'+media)
            pic = child(tree, P, 'pic'); nv = child(pic, P, 'nvPicPr')
            child(nv, P, 'cNvPr', id=idx, name=str(path.relative_to(OUT)))
            child(nv, P, 'cNvPicPr'); child(nv, P, 'nvPr')
            fill = child(pic, P, 'blipFill'); blip = child(fill, A, 'blip'); blip.set(f'{{{R}}}embed', f'rId{idx}')
            stretch = child(fill, A, 'stretch'); child(stretch, A, 'fillRect')
            props = child(pic, P, 'spPr'); xf = child(props, A, 'xfrm')
            child(xf, A, 'off', x=x*scale, y=y*scale); child(xf, A, 'ext', cx=w*scale, cy=h*scale)
            geom = child(props, A, 'prstGeom', prst='rect'); child(geom, A, 'avLst')
        cm = child(sld, P, 'clrMapOvr'); child(cm, A, 'masterClrMapping')
        files[f'ppt/slides/slide{slide_index}.xml'] = xml(sld)
        files[f'ppt/slides/_rels/slide{slide_index}.xml.rels'] = xml(slide_rels)
        entry = child(slides, P, 'sldId', id=255+slide_index); entry.set(f'{{{R}}}id', f'rId{20+slide_index}')
        child(rels, REL, 'Relationship', Id=f'rId{20+slide_index}', Type=R+'/slide', Target=f'slides/slide{slide_index}.xml')
        child(content_types, CT, 'Override', PartName=f'/ppt/slides/slide{slide_index}.xml', ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
        layouts.append(dict(environment=label, left='scene_focus.png', source_crop_pixels=crop,
                            sensor_rows=manifest['selected_sensors'], column_order=['RGB viewpoint', 'dense simulated depth', 'native simulated 8x8'],
                            panels=[dict(file=str(p.relative_to(OUT)), x=x, y=y, width=w, height=h) for p,x,y,w,h in panels]))
    files['ppt/presentation.xml'] = xml(presentation)
    files['ppt/_rels/presentation.xml.rels'] = xml(rels)
    files['[Content_Types].xml'] = xml(content_types)
    # Replace old document metadata, which described the earlier point-cloud deck.
    core = ET.fromstring(files['docProps/core.xml'])
    for element in core:
        if element.tag.endswith('}title'): element.text = 'Whole-body sensing: original scene assets'
        elif element.tag.endswith('}description'): element.text = 'Three text-free figure layouts; every image panel is independently editable.'
    files['docProps/core.xml'] = xml(core)
    with Image.open(OUT/'first_page_kitchen_items.png') as im:
        im.thumbnail((512, 512)); thumb = BytesIO(); im.convert('RGB').save(thumb, format='JPEG')
        files['docProps/thumbnail.jpeg'] = thumb.getvalue()
    with ZipFile(OUT/'whole_body_canva.pptx', 'w', ZIP_DEFLATED) as z:
        for name, data in files.items(): z.writestr(name, data)
    (OUT/'figure_layout.json').write_text(json.dumps(layouts, indent=2))
    gallery = OUT/'index.html'
    html = gallery.read_text()
    navigation = ('<p id="editable-layouts"><a href="whole_body_canva.pptx">Editable Canva PPTX</a> · '
                  '<a href="first_page_kitchen_items.png">Kitchen figure</a> · '
                  '<a href="first_page_cabinet_cavity.png">Cabinet figure</a> · '
                  '<a href="first_page_fumehood_clutter.png">Fume-hood figure</a> · '
                  '<a href="whole_body_figure_assets.zip">Download all images</a></p>')
    if 'id="editable-layouts"' not in html:
        html = html.replace('</h1>', '</h1>'+navigation, 1)
        gallery.write_text(html)
    with ZipFile(OUT/'whole_body_figure_assets.zip', 'w', ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and path.suffix not in ['.zip', '.mjb'] and 'source_snapshots' not in path.parts:
                z.write(path, path.relative_to(OUT))
    print('Saved three figure PNGs and editable Canva PPTX.')


if __name__ == '__main__': main()
