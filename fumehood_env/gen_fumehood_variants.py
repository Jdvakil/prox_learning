"""Emit fume-hood scene variants that differ in interior width, depth and height.

MuJoCo fixes geom sizes at compile time (the static BVH is built then), so hood
dimensions cannot be re-sized through mocap the way the sash and jambs are.
Instead each size lands in its own XML and the config hands them out per house
index via scene_xml_paths. Clutter and the place tray stay mocap: they are
re-posed every episode by the sampler.

Run it from anywhere; output lands next to this file in custom_scenes/. The 27
generated XMLs (3 widths x 3 depths x 3 heights) plus their metadata sidecars
are also committed, so running this is only needed to change the grid.
"""
import itertools
from pathlib import Path

OUT = Path(__file__).resolve().parent / "custom_scenes"
X_FRONT, Z_BENCH, T = 0.58, 0.72, 0.012
N_CLUTTER = 12

HEAD = """<mujoco model="{name}">
  <compiler angle="radian" autolimits="true" boundmass="0" balanceinertia="true"/>
  <option impratio="10" gravity="0 0 -9.8" integrator="implicitfast" cone="elliptic" jacobian="sparse" noslip_iterations="4">
    <flag multiccd="enable" contact="enable" warmstart="enable"/>
  </option>
  <size njmax="8000" nconmax="8000"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map znear="0.001"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.65 0.65 0.65"/>
  </visual>
  <asset>
    <texture name="benchtex" type="2d" builtin="flat" rgb1="0.85 0.86 0.88" width="32" height="32"/>
    <material name="bench" texture="benchtex" reflectance="0.1"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.46 0.46 0.48" rgb2="0.40 0.40 0.42" width="300" height="300"/>
    <material name="floormat" texture="floortex" texrepeat="10 10" reflectance="0.05"/>
  </asset>
  <default>
    <default class="hood">
      <geom contype="8" conaffinity="15" group="0" friction="0.9 0.9 0.001"
            solref="0.02 1" solimp="0.998 0.998 0.001" density="700" rgba="0.92 0.93 0.94 1"/>
    </default>
  </default>
  <worldbody>
    <light name="l_key" pos="0.4 0.5 2.6" dir="0 0 -1" diffuse="0.95 0.95 0.95"/>
    <light name="l_fill" pos="-0.6 -0.7 2.1" dir="0.3 0.3 -1" diffuse="0.5 0.5 0.5"/>
    <light name="l_hood" pos="{hx} 0 {lz}" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" material="floormat" contype="8" conaffinity="15"/>
    <geom name="room_wall_back"  class="hood" type="box" pos="-2.0 0 1.4" size="0.05 2.05 1.4" rgba="0.82 0.83 0.85 1"/>
    <geom name="room_wall_front" class="hood" type="box" pos="2.8 0 1.4"  size="0.05 2.05 1.4" rgba="0.82 0.83 0.85 1"/>
    <geom name="room_wall_left"  class="hood" type="box" pos="0.2 2.0 1.4"  size="2.65 0.05 1.4" rgba="0.80 0.81 0.84 1"/>
    <geom name="room_wall_right" class="hood" type="box" pos="0.2 -2.0 1.4" size="2.65 0.05 1.4" rgba="0.80 0.81 0.84 1"/>
    <geom name="room_ceiling"    class="hood" type="box" pos="0.2 0 2.82"  size="2.65 2.05 0.04" rgba="0.90 0.90 0.92 1"/>
"""

BODY = """    <geom name="bench_top"  class="hood" type="box" pos="{bx} 0 {bt}" size="{bl} {bw} 0.02" material="bench"/>
    <geom name="bench_body" class="hood" type="box" pos="{bx} 0 0.35" size="{bl2} {bw2} 0.33" rgba="0.80 0.81 0.83 1"/>
    <geom name="hood_side_l" class="hood" type="box" pos="{hx} {wy} {hz}" size="{hl} {t} {hh}" rgba="0.90 0.91 0.92 1"/>
    <geom name="hood_side_r" class="hood" type="box" pos="{hx} -{wy} {hz}" size="{hl} {t} {hh}" rgba="0.90 0.91 0.92 1"/>
    <geom name="hood_back"  class="hood" type="box" pos="{bkx} 0 {hz}" size="{t} {by} {hh}" rgba="0.88 0.89 0.90 1"/>
    <geom name="hood_top"   class="hood" type="box" pos="{hx} 0 {tz}" size="{tl} {by} 0.015"/>
    <geom name="hood_frame_l" class="hood" type="box" pos="{fx} {wy} {hz}" size="0.02 0.02 {hh}"/>
    <geom name="hood_frame_r" class="hood" type="box" pos="{fx} -{wy} {hz}" size="0.02 0.02 {hh}"/>
    <body name="sash" mocap="true" pos="{fx} 0 {sz}">
      <geom name="sash_g" class="hood" type="box" size="0.015 {by} 0.025" rgba="0.62 0.64 0.66 1"/>
    </body>
    <body name="jamb_l" mocap="true" pos="{fx} 0.35 {jz}">
      <geom name="jamb_l_g" class="hood" type="box" size="0.012 0.18 0.20" rgba="0.88 0.89 0.90 1"/>
    </body>
    <body name="jamb_r" mocap="true" pos="{fx} -0.35 {jz}">
      <geom name="jamb_r_g" class="hood" type="box" size="0.012 0.18 0.20" rgba="0.88 0.89 0.90 1"/>
    </body>
    <!-- place target for the pick-and-place variant: a shallow tray the arm sets
         the object down on. Mocap so the sampler can position it per episode. -->
    <body name="place_tray" mocap="true" pos="{ptx} {pty} {ptz}">
      <geom name="place_tray_g" class="hood" type="box" size="0.09 0.09 0.008" rgba="0.25 0.45 0.75 1"/>
    </body>
    <body name="protr_s" mocap="true" pos="0 0.8 -2.0">
      <geom name="protr_s_g" class="hood" type="box" size="0.0175 0.0175 0.10" rgba="0.55 0.6 0.65 1"/>
    </body>
    <body name="protr_m" mocap="true" pos="0 1.2 -2.0">
      <geom name="protr_m_g" class="hood" type="box" size="0.025 0.025 0.11" rgba="0.5 0.55 0.6 1"/>
    </body>
    <body name="protr_l" mocap="true" pos="0 1.6 -2.0">
      <geom name="protr_l_g" class="hood" type="box" size="0.035 0.035 0.12" rgba="0.45 0.5 0.55 1"/>
    </body>
"""

CLUTTER = """    <body name="cl_{i}" mocap="true" pos="{px} {py} -2.0">
      <geom name="cl_{i}_g" class="hood" type="{typ}" size="{sz}" rgba="{col} 1"/>
    </body>
"""
SHAPES = [("box", "0.030 0.030 0.075", "0.60 0.45 0.30"),
          ("cylinder", "0.032 0.055", "0.35 0.55 0.65"),
          ("box", "0.045 0.045 0.040", "0.70 0.65 0.35"),
          ("cylinder", "0.022 0.090", "0.30 0.60 0.45")]


def variant(name, half_w, depth, height):
    hx = X_FRONT + depth / 2
    hz = Z_BENCH + height / 2
    parts = [HEAD.format(name=name, hx=round(hx, 4), lz=round(Z_BENCH + height + 0.05, 4))]
    parts.append(BODY.format(
        bx=round(hx, 4), bt=round(Z_BENCH - 0.02, 4), bl=round(depth / 2 + 0.06, 4),
        bw=round(half_w + 0.10, 4), bl2=round(depth / 2 + 0.03, 4), bw2=round(half_w + 0.07, 4),
        hx=round(hx, 4), wy=round(half_w + T, 4), hz=round(hz, 4), hl=round(depth / 2, 4),
        t=T, hh=round(height / 2, 4), bkx=round(X_FRONT + depth + T, 4),
        by=round(half_w + 2 * T, 4), tz=round(Z_BENCH + height + 0.015, 4),
        tl=round(depth / 2 + T, 4), fx=X_FRONT,
        ptx=round(X_FRONT + 0.14, 4), pty=round(-(half_w - 0.12), 4),
        ptz=round(Z_BENCH + 0.008, 4),
        sz=round(Z_BENCH + height - 0.06, 4), jz=round(Z_BENCH + 0.20, 4)))
    for i in range(N_CLUTTER):
        typ, sz, col = SHAPES[i % len(SHAPES)]
        parts.append(CLUTTER.format(i=i, px=round(-1.0 - 0.2 * i, 3), py=3.0,
                                    typ=typ, sz=sz, col=col))
    parts.append("  </worldbody>\n</mujoco>\n")
    return "".join(parts)


WIDTHS = [0.32, 0.45, 0.58]    # interior HALF-widths (m)
DEPTHS = [0.58, 0.78, 1.00]    # interior depths (m)
HEIGHTS = [0.52, 0.81, 1.05]   # interior heights (m)
combos = list(itertools.product(WIDTHS, DEPTHS, HEIGHTS))

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for i, (w, d, h) in enumerate(combos):
        name = f"fumehood_v{i:02d}"
        (OUT / f"{name}.xml").write_text(variant(name, w, d, h))
        # every custom scene needs its metadata sidecar or scene compilation fails
        (OUT / f"{name}_metadata.json").write_text('{"objects": {}}\n')
        print(f"  {name}.xml   half_w={w}  depth={d}  height={h}")
    print(f"\nwrote {len(combos)} variants to {OUT}")
