"""
Generate GPRMax .in simulation files for oil-leak B-scan classifier training.

Two classes:
  label=1  leak    – PE bag with cooking oil buried in soil
  label=0  noleak  – three sub-types, all without oil:
              bare   : soil only, no target (12 soil permittivity variants)
              rock   : soil + rock/stone clutter (hyperbola, but no oil)
              pipe   : soil + empty PVC pipe  (hyperbola, but no oil)

File naming:
  leak_{bag_type}_{soil_key}_d{depth_cm:03d}_x{cx_cm:02d}.in
  noleak_bare_{soil_key}.in
  noleak_rock_{size}_{soil_key}_d{depth_cm:03d}_x{cx_cm:02d}.in
  noleak_pipe_{soil_key}_d{depth_cm:03d}_x{cx_cm:02d}.in

Run:
  python generate_cases.py          # generate all files
  python generate_cases.py --list   # preview counts without writing
"""

import os
import itertools
import argparse

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

DX = DY = DZ = 0.003

DOMAIN_X = 0.6
SOIL_TOP = 1.500
DOMAIN_Y = 1.533
DOMAIN_Z = 0.003
Z_MID    = DZ / 2

ANT_Y = SOIL_TOP - DY  # ground-coupled: antenna in last soil cell (cell 499, y=1.497)

TIME_WINDOW = 30e-9
PML  = 10
FREQ = 1.61e9

OFFSET     = 0.1
SCAN_START = 0.05
SCAN_STEP  = 0.02
N_SCANS    = 21

MAX_DEPTH_BOTTOM = SOIL_TOP

# ── Soil types used in LEAK scenarios (6 types, evenly spaced across εr range) ─
SOIL_TYPES = {
    "soil_s04": ( 4.5, 0.0005),   # very dry sand
    "soil_dry": ( 5.0, 0.001),    # dry soil
    "soil_s08": ( 8.0, 0.004),    # dry-medium
    "soil_med": (10.0, 0.008),    # medium
    "soil_s14": (14.0, 0.015),    # medium-wet
    "soil_wet": (18.0, 0.020),    # wet soil
}

# ── Bare-soil NOLEAK variants: continuous εr sweep ───────────────────────────
# 12 different soil permittivities → 12 noleak_bare_* files
BARE_SOIL_VARIANTS = [
    ("s03", 3.0,  0.0002),
    ("s04", 4.5,  0.0005),
    ("s06", 6.0,  0.001),
    ("s08", 8.0,  0.004),
    ("s10", 10.0, 0.008),
    ("s12", 12.0, 0.012),
    ("s14", 14.0, 0.015),
    ("s16", 16.0, 0.018),
    ("s18", 18.0, 0.020),
    ("s20", 20.0, 0.025),
    ("s22", 22.0, 0.030),
    ("s25", 25.0, 0.040),
]

# ── Clutter objects for NOLEAK (create hyperbolas but no oil) ─────────────────
# Rock: εr=6, σ=0.001 – common stone/gravel
# PVC pipe (empty air core): εr=3.0, σ=0.0001

ROCK_SIZES = {
    "sm": (0.030, 0.030),   # small stone  30×30 mm
    "md": (0.080, 0.050),   # medium rock  80×50 mm
}

PIPE_OD   = 0.10   # outer diameter of PVC pipe (m)
PIPE_WALL = 0.006  # wall thickness (m)

CLUTTER_DEPTHS  = [0.10, 0.25, 0.45, 0.65, 0.85, 1.05]   # m below surface
CLUTTER_CX_LIST = [0.15, 0.22, 0.30, 0.38, 0.45]          # m horizontal centre

# Soil types used in clutter – automatically follows SOIL_TYPES
CLUTTER_SOILS = list(SOIL_TYPES.keys())

# ── Bag types ──────────────────────────────────────────────────────────────────
BAG_TYPES = {
    "small_flat": (0.17,  0.025, "flat"),
    "small_vert": (0.025, 0.17,  "vert"),
    "large_flat": (0.22,  0.025, "flat"),
    "large_vert": (0.025, 0.15,  "vert"),
}

DEPTHS  = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.15]
CX_LIST = [0.15, 0.21, 0.27, 0.33, 0.40, 0.45]


# ── Helpers ───────────────────────────────────────────────────────────────────

def snap(v):
    return round(round(v / DX) * DX, 6)


def box(x1, y1, x2, y2, material):
    return f"#box: {x1} {y1} 0 {x2} {y2} {DOMAIN_Z} {material}"


def _base_header(title, er, sigma, extra_materials=None):
    L = [
        f"#title: {title}",
        "",
        f"#domain: {DOMAIN_X} {DOMAIN_Y} {DOMAIN_Z}",
        f"#dx_dy_dz: {DX} {DY} {DZ}",
        f"#time_window: {TIME_WINDOW}",
        "",
        f"#pml_cells: {PML} {PML} 0 {PML} 0 0",
        "",
        "## Materials",
        f"#material: {er} {sigma} 1 0 soil",
    ]
    if extra_materials:
        L += extra_materials
    L += [
        "",
        "## Domain fill",
        box(0, 0, DOMAIN_X, SOIL_TOP, "soil"),
        box(0, SOIL_TOP, DOMAIN_X, DOMAIN_Y, "free_space"),
        "",
    ]
    return L


def _antenna():
    return [
        f"#waveform: gaussian 1 {FREQ} mypulse",
        "",
        "## Ground-coupled sweep: run with -n 21",
        f"#hertzian_dipole: z {SCAN_START} {ANT_Y} {Z_MID} mypulse",
        f"#rx: {round(SCAN_START + OFFSET, 6)} {ANT_Y} {Z_MID}",
        f"#src_steps: {SCAN_STEP} 0 0",
        f"#rx_steps: {SCAN_STEP} 0 0",
        "",
    ]


# ── File generators ───────────────────────────────────────────────────────────

def make_noleak_bare(soil_key, er, sigma):
    L = _base_header(f"NOLEAK bare {soil_key} er={er}", er, sigma)
    L += _antenna()
    return "\n".join(L)


def make_noleak_rock(size_key, soil_key, er, sigma, depth, cx):
    rw, rh = ROCK_SIZES[size_key]
    rx1 = snap(cx - rw / 2);  rx2 = snap(cx + rw / 2)
    ry2 = snap(SOIL_TOP - depth)
    ry1 = snap(ry2 - rh)

    L = _base_header(
        f"NOLEAK rock_{size_key} {soil_key} depth={depth:.2f}m cx={cx:.2f}m",
        er, sigma,
        extra_materials=["#material: 6.0 0.001 1 0 rock"],
    )
    L += [
        f"## Clutter: rock_{size_key}  depth={depth:.2f}m  cx={cx:.2f}m",
        box(rx1, ry1, rx2, ry2, "rock"),
        "",
    ]
    L += _antenna()
    return "\n".join(L)


def make_noleak_pipe(soil_key, er, sigma, depth, cx):
    od = PIPE_OD
    t  = PIPE_WALL
    px1 = snap(cx - od / 2);  px2 = snap(cx + od / 2)
    py2 = snap(SOIL_TOP - depth)
    py1 = snap(py2 - od)
    # air core (overwrite PVC interior with free_space)
    ix1 = snap(px1 + t);  ix2 = snap(px2 - t)
    iy1 = snap(py1 + t);  iy2 = snap(py2 - t)

    L = _base_header(
        f"NOLEAK pipe {soil_key} depth={depth:.2f}m cx={cx:.2f}m",
        er, sigma,
        extra_materials=["#material: 3.0 0.0001 1 0 pvc"],
    )
    L += [
        f"## Clutter: empty PVC pipe  depth={depth:.2f}m  cx={cx:.2f}m",
        box(px1, py1, px2, py2, "pvc"),
        "## Air core",
        box(ix1, iy1, ix2, iy2, "free_space"),
        "",
    ]
    L += _antenna()
    return "\n".join(L)


def make_leak_file(bag_type, er, sigma, depth, cx):
    bag_x, bag_z, _ = BAG_TYPES[bag_type]
    by2 = snap(SOIL_TOP - depth)
    by1 = snap(SOIL_TOP - depth - bag_z)
    bx1 = snap(cx - bag_x / 2);  bx2 = snap(cx + bag_x / 2)
    oy1 = snap(by1 + DY);  oy2 = snap(by2 - DY)
    ox1 = snap(bx1 + DX);  ox2 = snap(bx2 - DX)

    L = _base_header(
        f"LEAK {bag_type} er={er} depth={depth:.2f}m cx={cx:.2f}m",
        er, sigma,
        extra_materials=[
            "#material: 2.3 0.0001 1 0 pe_wall",
            "#material: 3.1 0.00005 1 0 cooking_oil",
        ],
    )
    L += [
        f"## Target: {bag_type}  depth={depth:.2f}m  cx={cx:.2f}m",
        box(bx1, by1, bx2, by2, "pe_wall"),
        box(ox1, oy1, ox2, oy2, "cooking_oil"),
        "",
    ]
    L += _antenna()
    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    manifest = []
    generated = skipped = 0

    def write(fname, content):
        nonlocal generated
        if not args.list:
            with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
                f.write(content)
            generated += 1

    def add(fname, label, bag_type="", soil="", depth_m="", cx_m="", clutter=""):
        manifest.append({
            "filename": fname, "label": label,
            "bag_type": bag_type, "soil": soil,
            "depth_m": depth_m, "cx_m": cx_m,
            "clutter": clutter,
        })

    # ── Noleak: bare soil (12 εr variants) ────────────────────────────────────
    for soil_key, er, sigma in BARE_SOIL_VARIANTS:
        fname = f"noleak_bare_{soil_key}.in"
        add(fname, 0, soil=soil_key, clutter="bare")
        write(fname, make_noleak_bare(soil_key, er, sigma))

    # ── Noleak: rock clutter ──────────────────────────────────────────────────
    for size_key, soil_key, depth, cx in itertools.product(
            ROCK_SIZES.keys(), CLUTTER_SOILS, CLUTTER_DEPTHS, CLUTTER_CX_LIST):
        er, sigma = SOIL_TYPES[soil_key]
        d_cm = int(round(depth * 100))
        x_cm = int(round(cx    * 100))
        fname = f"noleak_rock{size_key}_{soil_key}_d{d_cm:03d}_x{x_cm:02d}.in"
        add(fname, 0, soil=soil_key, depth_m=depth, cx_m=cx, clutter=f"rock_{size_key}")
        write(fname, make_noleak_rock(size_key, soil_key, er, sigma, depth, cx))

    # ── Noleak: empty PVC pipe ────────────────────────────────────────────────
    for soil_key, depth, cx in itertools.product(
            CLUTTER_SOILS, CLUTTER_DEPTHS, CLUTTER_CX_LIST):
        er, sigma = SOIL_TYPES[soil_key]
        d_cm = int(round(depth * 100))
        x_cm = int(round(cx    * 100))
        fname = f"noleak_pipe_{soil_key}_d{d_cm:03d}_x{x_cm:02d}.in"
        add(fname, 0, soil=soil_key, depth_m=depth, cx_m=cx, clutter="pipe")
        write(fname, make_noleak_pipe(soil_key, er, sigma, depth, cx))

    # ── Leak: bag + oil ───────────────────────────────────────────────────────
    for bag_type, soil_key, depth, cx in itertools.product(
            BAG_TYPES.keys(), SOIL_TYPES.keys(), DEPTHS, CX_LIST):
        _, bag_z, _ = BAG_TYPES[bag_type]
        if depth + bag_z >= MAX_DEPTH_BOTTOM:
            skipped += 1
            continue
        er, sigma = SOIL_TYPES[soil_key]
        d_cm = int(round(depth * 100))
        x_cm = int(round(cx    * 100))
        fname = f"leak_{bag_type}_{soil_key}_d{d_cm:03d}_x{x_cm:02d}.in"
        add(fname, 1, bag_type=bag_type, soil=soil_key, depth_m=depth, cx_m=cx)
        write(fname, make_leak_file(bag_type, er, sigma, depth, cx))

    # ── Summary ───────────────────────────────────────────────────────────────
    n0 = sum(1 for m in manifest if m["label"] == 0)
    n1 = sum(1 for m in manifest if m["label"] == 1)
    n_bare  = sum(1 for m in manifest if m["clutter"] == "bare")
    n_rock  = sum(1 for m in manifest if "rock" in m["clutter"])
    n_pipe  = sum(1 for m in manifest if m["clutter"] == "pipe")

    if args.list:
        for m in manifest:
            print(m["filename"])

    print(f"\n{'─'*55}")
    print(f"noleak 合计: {n0}  (bare={n_bare}  rock={n_rock}  pipe={n_pipe})")
    print(f"leak   合计: {n1}  (skipped {skipped} depth-overflow)")
    print(f"总计:        {n0 + n1}")
    print(f"类别比例:    1 : {n1/n0:.1f}  →  训练时建议 class_weight={{0:{n1//n0}, 1:1}}")

    if not args.list:
        print(f"已写入:      {generated} 个文件  →  {OUTPUT_DIR}")

    return manifest


if __name__ == "__main__":
    main()
