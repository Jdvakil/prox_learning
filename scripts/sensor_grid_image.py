"""Draw crisp cell boundaries on enlarged sensor images without changing data."""
from pathlib import Path
import json
import zipfile

from PIL import Image, ImageDraw


def bordered_sensor_grid(native, size=1024, line_width=4):
    """Nearest-neighbor enlargement with black cell lines and an outer border."""
    if native.size != (8, 8):
        raise ValueError('Expected an 8x8 sensor image')
    grid = native.convert('RGB').resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(grid)
    for i in range(1, 8):
        coordinate = round(i * size / 8)
        draw.line((coordinate, 0, coordinate, size - 1), fill='black', width=line_width)
        draw.line((0, coordinate, size - 1, coordinate), fill='black', width=line_width)
    draw.rectangle((0, 0, size - 1, size - 1), outline='black', width=line_width)
    return grid


def refresh_existing_exports():
    root = Path(__file__).resolve().parents[1] / 'images/table_occlusion'
    for folder, archive_name in [('sensor_views', 'sensor_rgb_depth_8x8.zip'),
                                  ('lowered_robot', 'lowered_robot_images.zip')]:
        directory = root / folder
        if not directory.exists():
            continue
        for sensor in sorted(directory.glob('link6_sensor_*')):
            with Image.open(sensor / 'readout_8x8_native.png') as native:
                bordered_sensor_grid(native).save(sensor / 'readout_8x8.png')
            sheet = Image.new('RGB', (3096, 1024), 'white')
            for j, filename in enumerate(['rgb.png', 'depth.png', 'readout_8x8.png']):
                with Image.open(sensor / filename) as panel:
                    sheet.paste(panel, (j * 1036, 0))
            sheet.save(sensor / 'rgb_depth_8x8.png')
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['grid_display'] = '1024x1024 nearest-neighbor display with solid black 4 px cell boundaries and outer border. Native 8x8 PNG and numerical arrays unchanged.'
        manifest['processing'] = manifest['processing'].replace('annotations, ', '')
        manifest_path.write_text(json.dumps(manifest, indent=2))
        with zipfile.ZipFile(directory / archive_name, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.rglob('*')):
                if path.is_file() and path.suffix != '.zip':
                    archive.write(path, path.relative_to(directory))
        print('Updated', directory)


if __name__ == '__main__':
    refresh_existing_exports()
