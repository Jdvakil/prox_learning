#!/usr/bin/python3
"""Export system-diagram SVGs using installed librsvg/Cairo, then package assets."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import cairo
import gi

gi.require_version('Rsvg', '2.0')
from gi.repository import Rsvg

OUT = Path(__file__).resolve().parents[1]/'images/system_diagram'


def main():
    for path in sorted(OUT.glob('*.svg')):
        handle = Rsvg.Handle.new_from_file(str(path))
        viewport = Rsvg.Rectangle()
        viewport.x, viewport.y, viewport.width, viewport.height = 0, 0, 1840, 760
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 3680, 1520)
        context = cairo.Context(surface); context.scale(2, 2)
        handle.render_document(context, viewport)
        surface.write_to_png(str(path.with_suffix('.png')))
        surface = cairo.PDFSurface(str(path.with_suffix('.pdf')), 920, 380)
        context = cairo.Context(surface); context.scale(.5, .5)
        handle.render_document(context, viewport); surface.finish()
    (OUT/'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>PACT system diagram</title>'
        '<style>body{font:17px Arial;max-width:1500px;margin:40px auto;line-height:1.5}img{width:100%}section{margin:50px 0}</style>'
        '<h1>PACT system diagram</h1><p><a href="system_diagram_canva.pptx">Editable Canva PPTX</a> · '
        '<a href="system_diagram_assets.zip">All formats</a></p>'
        '<p>The PPTX contains separate editable boxes, arrows, labels, and original-image thumbnails. '
        'SVG and PDF exports retain vector diagram elements.</p>'
        + ''.join(f'<section><h2>{title}</h2><img src="{stem}.png"><p>'
                  f'<a href="{stem}.svg">SVG</a> · <a href="{stem}.pdf">PDF</a> · '
                  f'<a href="{stem}.png">PNG</a></p></section>'
                  for stem,title in [('system_overview','System overview'),
                                     ('readout_architecture','Readout architecture'),
                                     ('system_overview_unlabelled','Unlabelled overview')]))
    with ZipFile(OUT/'system_diagram_assets.zip', 'w', ZIP_DEFLATED) as archive:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.suffix != '.zip': archive.write(path, path.name)
    print('Exported PNG/PDF previews and complete diagram bundle.')


if __name__ == '__main__': main()
