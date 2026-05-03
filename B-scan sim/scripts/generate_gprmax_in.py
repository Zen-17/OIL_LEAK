#!/usr/bin/env python
"""Generate varied realistic gprMax .in files for shallow oil-leak detection."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

POS_SCENARIOS = (
    "plume_diffuse",
    "pool_capillary",
    "layered_sheet",
    "pocket_chain",
)

NEG_SCENARIOS = (
    "wet_lens",
    "clay_lens",
    "buried_pipe",
    "mixed_clutter",
)

ALL_SCENARIOS_REALISTIC = POS_SCENARIOS + NEG_SCENARIOS

POS_SCENARIOS_SANDBOX = (
    "sandbox_local_patch",
    "sandbox_local_irregular",
    "sandbox_local_pool",
)

NEG_SCENARIOS_SANDBOX = (
    "sandbox_clean",
    "sandbox_wet_patch",
    "sandbox_shallow_clay",
)

PROFILES = ("realistic", "sandbox_local_easy")
ALL_SCENARIOS = ALL_SCENARIOS_REALISTIC + POS_SCENARIOS_SANDBOX + NEG_SCENARIOS_SANDBOX


def _rand(rng: random.Random, lo: float, hi: float) -> float:
    return lo + (hi - lo) * rng.random()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _box_specs_to_lines(specs: List[Tuple[float, float, float, float, str]]) -> List[str]:
    lines: List[str] = []
    for x0, y0, x1, y1, material in specs:
        lines.append(f"#box: {x0:.3f} {y0:.3f} 0.000 {x1:.3f} {y1:.3f} 0.002 {material}")
    return lines


def _transform_box_specs(
    specs: List[Tuple[float, float, float, float, str]],
    x_ref: float,
    tilt_deg: float,
    mirror_lr: bool,
    y_min: float = 0.465,
    y_max: float = 0.676,
) -> List[Tuple[float, float, float, float, str]]:
    out: List[Tuple[float, float, float, float, str]] = []
    tilt_slope = math.tan(math.radians(tilt_deg))

    for x0, y0, x1, y1, material in specs:
        if mirror_lr:
            x0, x1 = 2.0 * x_ref - x1, 2.0 * x_ref - x0

        xc = 0.5 * (x0 + x1)
        dy = tilt_slope * (xc - x_ref)
        y0n = y0 + dy
        y1n = y1 + dy

        if y0n < y_min:
            shift = y_min - y0n
            y0n += shift
            y1n += shift
        if y1n > y_max:
            shift = y1n - y_max
            y0n -= shift
            y1n -= shift

        y0n = _clamp(y0n, y_min, y_max - 0.004)
        y1n = _clamp(y1n, y0n + 0.003, y_max)

        x0n = _clamp(x0, 0.02, 1.18)
        x1n = _clamp(x1, 0.02, 1.18)
        if x1n <= x0n + 0.006:
            xmid = 0.5 * (x0n + x1n)
            x0n = _clamp(xmid - 0.003, 0.02, 1.18)
            x1n = _clamp(xmid + 0.003, 0.02, 1.18)

        out.append((x0n, y0n, x1n, y1n, material))

    return out


def _sandbox_clutter(
    rng: random.Random,
    clutter_level: float,
    avoid_x: float | None,
) -> Tuple[List[str], Dict[str, float | int]]:
    meta: Dict[str, float | int] = {
        "sandbox_clutter_level": round(clutter_level, 3),
        "sandbox_pipe_count": 0,
        "sandbox_stone_count": 0,
    }
    if clutter_level <= 0:
        return [], meta

    lines: List[str] = []
    gap_x = 0.10

    def _sample_x(x_lo: float, x_hi: float, avoid: float | None, gap: float) -> float:
        if avoid is None:
            return _rand(rng, x_lo, x_hi)
        for _ in range(12):
            x = _rand(rng, x_lo, x_hi)
            if abs(x - avoid) >= gap:
                return x
        return _rand(rng, x_lo, x_hi)

    pipe_prob = 0.18 + 0.55 * clutter_level
    if rng.random() < pipe_prob:
        # Pipe clutter is compact and curve-dominant in B-scan, distinct from oil patches.
        x = _sample_x(0.10, 1.10, avoid_x, gap_x)
        y = _rand(rng, 0.490, 0.660)
        r = _rand(rng, 0.009, 0.022)
        lines.append(f"#cylinder: {x:.3f} {y:.3f} 0.000 {x:.3f} {y:.3f} 0.002 {r:.3f} pvc")
        meta["sandbox_pipe_count"] = 1

    max_stones = 1 + int(round(3 * clutter_level))
    n_stones = rng.randint(1, max_stones)
    meta["sandbox_stone_count"] = n_stones
    for _ in range(n_stones):
        x_center = _sample_x(0.10, 1.10, avoid_x, gap_x)
        w = _rand(rng, 0.028, 0.060)
        x0 = _clamp(x_center - 0.5 * w, 0.08, 1.12)
        y0 = _rand(rng, 0.495, 0.655)
        h = _rand(rng, 0.014, 0.040)
        x1 = _clamp(x0 + w, 0.08, 1.16)
        y1 = _clamp(y0 + h, 0.505, 0.674)
        lines.append(f"#box: {x0:.3f} {y0:.3f} 0.000 {x1:.3f} {y1:.3f} 0.002 gravel")

    return lines, meta


def _surface_boxes(
    rng: random.Random,
    x_min: float,
    x_max: float,
    y_top: float,
    segments: int,
    rough_min: float = -0.007,
    rough_max: float = 0.005,
    min_thickness: float = 0.004,
    max_thickness: float = 0.020,
) -> List[str]:
    lines: List[str] = []
    dx = (x_max - x_min) / segments
    x0 = x_min
    for _ in range(segments):
        x1 = x0 + dx
        rough = _rand(rng, rough_min, rough_max)
        mean_thickness = 0.5 * (min_thickness + max_thickness)
        y0 = max(
            y_top - max_thickness,
            min(y_top - min_thickness, y_top + rough - mean_thickness),
        )
        lines.append(
            f"#box: {x0:.3f} {y0:.3f} 0.000 {x1:.3f} {y_top:.3f} 0.002 dry_topsoil"
        )
        x0 = x1
    return lines


def _common_clutter(rng: random.Random) -> List[str]:
    x_left = _rand(rng, 0.08, 0.18)
    x_right = _rand(rng, 0.92, 1.05)
    return [
        f"#box: {x_left:.3f} 0.335 0.000 {x_left + 0.14:.3f} 0.430 0.002 wet_sand",
        f"#box: {x_right:.3f} 0.315 0.000 {x_right + 0.13:.3f} 0.405 0.002 wet_sand",
        "#box: 0.300 0.270 0.000 0.365 0.335 0.002 gravel",
        "#box: 0.900 0.245 0.000 0.975 0.320 0.002 gravel",
        f"#cylinder: {_rand(rng, 0.64, 0.82):.3f} 0.305 0.000 {_rand(rng, 0.64, 0.82):.3f} 0.305 0.002 {_rand(rng, 0.018, 0.032):.3f} pvc",
    ]


def _positive_geometry(rng: random.Random, scenario: str) -> Tuple[List[str], Dict[str, float | str]]:
    meta: Dict[str, float | str] = {"scenario": scenario}

    if scenario == "plume_diffuse":
        cx = _rand(rng, 0.50, 0.74)
        y_mid = _rand(rng, 0.49, 0.57)
        w = _rand(rng, 0.20, 0.34)
        h = _rand(rng, 0.08, 0.14)
        lines = [
            f"#box: {cx - 1.20*w:.3f} {y_mid - 0.60*h:.3f} 0.000 {cx + 1.05*w:.3f} {y_mid + 0.50*h:.3f} 0.002 oil_saturated_sand",
            f"#box: {cx - 0.95*w:.3f} {y_mid - 0.15*h:.3f} 0.000 {cx + 0.35*w:.3f} {y_mid + 0.95*h:.3f} 0.002 oil_saturated_sand",
            f"#box: {cx - 0.12*w:.3f} {y_mid - 0.82*h:.3f} 0.000 {cx + 1.18*w:.3f} {y_mid + 0.22*h:.3f} 0.002 oil_saturated_sand",
            f"#box: {cx - 0.24*w:.3f} {y_mid - 0.12*h:.3f} 0.000 {cx + 0.08*w:.3f} {y_mid + 0.30*h:.3f} 0.002 free_oil",
            f"#box: {cx + 0.28*w:.3f} {y_mid - 0.30*h:.3f} 0.000 {cx + 0.52*w:.3f} {y_mid + 0.05*h:.3f} 0.002 free_oil",
        ]
        meta.update({"oil_center_x": round(cx, 3), "oil_center_y": round(y_mid, 3)})
        return lines, meta

    if scenario == "pool_capillary":
        x0 = _rand(rng, 0.30, 0.42)
        x1 = _rand(rng, 0.78, 0.96)
        y0 = _rand(rng, 0.44, 0.49)
        y1 = _rand(rng, 0.56, 0.62)
        lines = [
            f"#box: {x0:.3f} {y0:.3f} 0.000 {x1:.3f} {y1:.3f} 0.002 oil_saturated_sand",
            f"#box: {x0 + 0.12:.3f} {y0 + 0.03:.3f} 0.000 {x1 - 0.10:.3f} {y1 - 0.03:.3f} 0.002 free_oil",
            f"#box: {x0 + 0.06:.3f} {y1 - 0.02:.3f} 0.000 {x0 + 0.28:.3f} {y1 + 0.05:.3f} 0.002 oil_saturated_sand",
            f"#box: {x1 - 0.30:.3f} {y1 - 0.02:.3f} 0.000 {x1 - 0.08:.3f} {y1 + 0.05:.3f} 0.002 oil_saturated_sand",
        ]
        meta.update({"pool_x0": round(x0, 3), "pool_x1": round(x1, 3)})
        return lines, meta

    if scenario == "layered_sheet":
        y_top = _rand(rng, 0.43, 0.49)
        thickness = _rand(rng, 0.05, 0.10)
        lines = [
            f"#box: 0.180 {y_top:.3f} 0.000 1.040 {y_top + thickness:.3f} 0.002 oil_saturated_sand",
            f"#box: 0.240 {y_top + 0.01:.3f} 0.000 0.500 {y_top + thickness + 0.04:.3f} 0.002 oil_saturated_sand",
            f"#box: 0.760 {y_top + 0.01:.3f} 0.000 0.980 {y_top + thickness + 0.05:.3f} 0.002 oil_saturated_sand",
            f"#box: 0.430 {y_top + 0.015:.3f} 0.000 0.560 {y_top + thickness - 0.01:.3f} 0.002 free_oil",
        ]
        meta.update({"sheet_top": round(y_top, 3), "sheet_thickness": round(thickness, 3)})
        return lines, meta

    if scenario == "pocket_chain":
        x_start = _rand(rng, 0.28, 0.36)
        y_base = _rand(rng, 0.47, 0.53)
        dx = _rand(rng, 0.10, 0.14)
        lines = []
        for i in range(5):
            px = x_start + i * dx
            py = y_base + _rand(rng, -0.04, 0.04)
            w = _rand(rng, 0.07, 0.12)
            h = _rand(rng, 0.04, 0.08)
            lines.append(
                f"#box: {px - w/2:.3f} {py - h/2:.3f} 0.000 {px + w/2:.3f} {py + h/2:.3f} 0.002 oil_saturated_sand"
            )
            if i % 2 == 1:
                lines.append(
                    f"#box: {px - 0.25*w:.3f} {py - 0.25*h:.3f} 0.000 {px + 0.20*w:.3f} {py + 0.20*h:.3f} 0.002 free_oil"
                )
        meta.update({"chain_start_x": round(x_start, 3), "chain_y": round(y_base, 3)})
        return lines, meta

    raise ValueError(f"Unsupported positive scenario: {scenario}")


def _negative_geometry(rng: random.Random, scenario: str) -> Tuple[List[str], Dict[str, float | str]]:
    meta: Dict[str, float | str] = {"scenario": scenario}

    if scenario == "wet_lens":
        lines = [
            "#box: 0.250 0.430 0.000 0.980 0.525 0.002 wet_sand",
            "#box: 0.420 0.470 0.000 0.760 0.605 0.002 wet_sand",
            "#box: 0.560 0.440 0.000 0.860 0.515 0.002 clay_lens",
        ]
        return lines, meta

    if scenario == "clay_lens":
        lines = [
            "#box: 0.300 0.435 0.000 0.980 0.525 0.002 clay_lens",
            "#box: 0.420 0.470 0.000 0.760 0.585 0.002 wet_sand",
            "#box: 0.500 0.430 0.000 0.640 0.500 0.002 clay_lens",
        ]
        return lines, meta

    if scenario == "buried_pipe":
        pipe_x = _rand(rng, 0.45, 0.85)
        pipe_r = _rand(rng, 0.024, 0.038)
        lines = [
            "#box: 0.220 0.440 0.000 0.960 0.515 0.002 wet_sand",
            "#box: 0.420 0.470 0.000 0.720 0.600 0.002 wet_sand",
            f"#cylinder: {pipe_x:.3f} 0.410 0.000 {pipe_x:.3f} 0.410 0.002 {pipe_r:.3f} pvc",
        ]
        meta.update({"pipe_x": round(pipe_x, 3), "pipe_r": round(pipe_r, 3)})
        return lines, meta

    if scenario == "mixed_clutter":
        lines = [
            "#box: 0.280 0.430 0.000 0.620 0.520 0.002 wet_sand",
            "#box: 0.620 0.445 0.000 0.980 0.535 0.002 clay_lens",
            "#box: 0.440 0.490 0.000 0.760 0.600 0.002 wet_sand",
            "#box: 0.350 0.350 0.000 0.430 0.420 0.002 gravel",
            "#box: 0.800 0.330 0.000 0.890 0.410 0.002 gravel",
        ]
        return lines, meta

    raise ValueError(f"Unsupported negative scenario: {scenario}")


def _positive_geometry_sandbox(
    rng: random.Random, scenario: str, tilt_max_deg: float
) -> Tuple[List[str], Dict[str, float | str]]:
    meta: Dict[str, float | str] = {"scenario": scenario}
    tilt_deg = _rand(rng, -tilt_max_deg, tilt_max_deg)
    mirror_lr = rng.random() < 0.5

    if scenario == "sandbox_local_patch":
        x_c = _rand(rng, 0.22, 0.98)
        width = _rand(rng, 0.20, 0.46)
        y_top = _rand(rng, 0.500, 0.650)
        thickness = _rand(rng, 0.032, 0.090)
        specs = [
            (x_c - width / 2, y_top, x_c + width / 2, y_top + thickness, "oil_saturated_sand"),
            (
                x_c - 0.26 * width,
                y_top + 0.008,
                x_c + 0.28 * width,
                y_top + 0.86 * thickness,
                "free_oil",
            ),
        ]
        lines = _box_specs_to_lines(
            _transform_box_specs(
                specs=specs,
                x_ref=x_c,
                tilt_deg=tilt_deg,
                mirror_lr=mirror_lr,
            )
        )
        meta.update(
            {
                "oil_center_x": round(x_c, 3),
                "oil_top_y": round(y_top, 3),
                "oil_tilt_deg": round(tilt_deg, 3),
                "oil_mirror_lr": int(mirror_lr),
            }
        )
        return lines, meta

    if scenario == "sandbox_local_irregular":
        x_c = _rand(rng, 0.22, 0.98)
        y_c = _rand(rng, 0.510, 0.662)
        w = _rand(rng, 0.16, 0.38)
        h = _rand(rng, 0.032, 0.100)
        specs = [
            (
                x_c - 0.60 * w,
                y_c - 0.50 * h,
                x_c + 0.75 * w,
                y_c + 0.35 * h,
                "oil_saturated_sand",
            ),
            (
                x_c - 0.80 * w,
                y_c - 0.20 * h,
                x_c + 0.10 * w,
                y_c + 0.65 * h,
                "oil_saturated_sand",
            ),
            (
                x_c + 0.15 * w,
                y_c - 0.35 * h,
                x_c + 0.40 * w,
                y_c + 0.10 * h,
                "free_oil",
            ),
            (
                x_c - 0.35 * w,
                y_c - 0.05 * h,
                x_c - 0.08 * w,
                y_c + 0.42 * h,
                "free_oil",
            ),
        ]
        lines = _box_specs_to_lines(
            _transform_box_specs(
                specs=specs,
                x_ref=x_c,
                tilt_deg=tilt_deg,
                mirror_lr=mirror_lr,
            )
        )
        meta.update(
            {
                "oil_center_x": round(x_c, 3),
                "oil_center_y": round(y_c, 3),
                "oil_tilt_deg": round(tilt_deg, 3),
                "oil_mirror_lr": int(mirror_lr),
            }
        )
        return lines, meta

    if scenario == "sandbox_local_pool":
        width = _rand(rng, 0.20, 0.52)
        x0 = _rand(rng, 0.14, 1.10 - width)
        x1 = x0 + width
        y0 = _rand(rng, 0.502, 0.654)
        y1 = _clamp(y0 + _rand(rng, 0.032, 0.094), y0 + 0.026, 0.676)
        x_ref = 0.5 * (x0 + x1)
        specs = [
            (x0, y0, x1, y1, "oil_saturated_sand"),
            (x0 + 0.06, y0 + 0.006, x1 - 0.06, y1 - 0.006, "free_oil"),
        ]
        lines = _box_specs_to_lines(
            _transform_box_specs(
                specs=specs,
                x_ref=x_ref,
                tilt_deg=tilt_deg,
                mirror_lr=mirror_lr,
            )
        )
        meta.update(
            {
                "pool_x0": round(x0, 3),
                "pool_x1": round(x1, 3),
                "oil_tilt_deg": round(tilt_deg, 3),
                "oil_mirror_lr": int(mirror_lr),
            }
        )
        return lines, meta

    raise ValueError(f"Unsupported sandbox positive scenario: {scenario}")


def _negative_geometry_sandbox(
    rng: random.Random, scenario: str, tilt_max_deg: float
) -> Tuple[List[str], Dict[str, float | str]]:
    meta: Dict[str, float | str] = {"scenario": scenario}
    tilt_deg = _rand(rng, -0.6 * tilt_max_deg, 0.6 * tilt_max_deg)
    mirror_lr = rng.random() < 0.5

    if scenario == "sandbox_clean":
        return [], meta

    if scenario == "sandbox_wet_patch":
        x_c = _rand(rng, 0.18, 1.02)
        w = _rand(rng, 0.07, 0.20)
        y_top = _rand(rng, 0.500, 0.662)
        y_bot = _clamp(y_top + _rand(rng, 0.012, 0.042), y_top + 0.010, 0.676)
        specs = [(x_c - w / 2, y_top, x_c + w / 2, y_bot, "wet_sand")]
        lines = _box_specs_to_lines(
            _transform_box_specs(
                specs=specs,
                x_ref=x_c,
                tilt_deg=tilt_deg,
                mirror_lr=mirror_lr,
            )
        )
        meta.update(
            {
                "wet_center_x": round(x_c, 3),
                "neg_tilt_deg": round(tilt_deg, 3),
                "neg_mirror_lr": int(mirror_lr),
            }
        )
        return lines, meta

    if scenario == "sandbox_shallow_clay":
        w = _rand(rng, 0.09, 0.26)
        x0 = _rand(rng, 0.14, 1.10 - w)
        x1 = x0 + w
        y0 = _rand(rng, 0.502, 0.664)
        y1 = _clamp(y0 + _rand(rng, 0.010, 0.040), y0 + 0.009, 0.676)
        x_ref = 0.5 * (x0 + x1)
        specs = [(x0, y0, x1, y1, "clay_lens")]
        lines = _box_specs_to_lines(
            _transform_box_specs(
                specs=specs,
                x_ref=x_ref,
                tilt_deg=tilt_deg,
                mirror_lr=mirror_lr,
            )
        )
        meta.update(
            {
                "neg_tilt_deg": round(tilt_deg, 3),
                "neg_mirror_lr": int(mirror_lr),
            }
        )
        return lines, meta

    raise ValueError(f"Unsupported sandbox negative scenario: {scenario}")


def _select_scenario(label: str, scenario: str, rng: random.Random, profile: str) -> str:
    if scenario == "auto":
        if profile == "sandbox_local_easy":
            return rng.choice(
                POS_SCENARIOS_SANDBOX if label == "oil" else NEG_SCENARIOS_SANDBOX
            )
        return rng.choice(POS_SCENARIOS if label == "oil" else NEG_SCENARIOS)

    if profile == "sandbox_local_easy":
        if label == "oil" and scenario in POS_SCENARIOS_SANDBOX:
            return scenario
        if label == "no_oil" and scenario in NEG_SCENARIOS_SANDBOX:
            return scenario
        raise ValueError(
            f"Scenario '{scenario}' does not match label/profile '{label}/{profile}'. "
            f"Oil: {POS_SCENARIOS_SANDBOX}; no_oil: {NEG_SCENARIOS_SANDBOX}."
        )

    if label == "oil" and scenario in POS_SCENARIOS:
        return scenario
    if label == "no_oil" and scenario in NEG_SCENARIOS:
        return scenario

    raise ValueError(
        f"Scenario '{scenario}' does not match label '{label}'. "
        f"Oil: {POS_SCENARIOS}; no_oil: {NEG_SCENARIOS}."
    )


def build_scene(
    label: str,
    freq_hz: float,
    scenario: str,
    seed: int | None,
    profile: str = "realistic",
    sandbox_tilt_max_deg: float = 10.0,
    sandbox_clutter_level: float = 0.35,
    simplify_water: bool = False,
) -> Tuple[str, Dict[str, float | str | int]]:
    rng = random.Random(seed)

    if profile not in PROFILES:
        raise ValueError(f"Unsupported profile '{profile}', choose from {PROFILES}")

    chosen_scenario = _select_scenario(
        label=label, scenario=scenario, rng=rng, profile=profile
    )

    y_surface = 0.680
    air_top = 0.800
    time_window = 42e-9
    pml_cells = "10 10 0 10 10 0"

    if profile == "sandbox_local_easy":
        if simplify_water:
            # Medium-low moisture band for clearer oil-vs-background contrast
            # while staying more realistic than ultra-dry settings.
            water_low = _rand(rng, 0.050, 0.075)
            water_high = _rand(rng, 0.090, 0.120)
            fractal_dim = _rand(rng, 1.26, 1.36)
            fractal_bins = rng.randint(26, 36)
        else:
            water_low = _rand(rng, 0.02, 0.04)
            water_high = _rand(rng, 0.05, 0.09)
            fractal_dim = _rand(rng, 1.25, 1.42)
            fractal_bins = rng.randint(24, 40)
        # Keep Tx/Rx almost touching the ground surface (slightly above interface).
        ant_height = _rand(rng, y_surface + 0.001, y_surface + 0.004)
        soil_density = _rand(rng, 1.56, 1.78)
        if simplify_water:
            dry_er = _rand(rng, 3.0, 3.6)
            dry_sigma = _rand(rng, 0.0008, 0.0016)
            moist_er = _rand(rng, 5.2, 6.6)
            moist_sigma = _rand(rng, 0.0028, 0.0058)
            wet_er = _rand(rng, 9.5, 11.8)
            wet_sigma = _rand(rng, 0.0080, 0.0135)
            clay_er = _rand(rng, 13.5, 16.5)
            clay_sigma = _rand(rng, 0.014, 0.022)
        else:
            dry_er = _rand(rng, 2.8, 3.5)
            dry_sigma = _rand(rng, 0.0005, 0.0012)
            moist_er = _rand(rng, 4.5, 6.0)
            moist_sigma = _rand(rng, 0.0015, 0.0045)
            wet_er = _rand(rng, 7.5, 10.5)
            wet_sigma = _rand(rng, 0.005, 0.013)
            clay_er = _rand(rng, 12.0, 17.0)
            clay_sigma = _rand(rng, 0.010, 0.024)
        y_surface = 0.680
        air_top = 0.700
        time_window = _rand(rng, 20e-9, 28e-9)
        # Keep top PML thinner for near-surface coupling so Tx/Rx are not inside PML.
        pml_cells = "10 10 0 10 4 0"
    else:
        water_low = _rand(rng, 0.04, 0.10)
        water_high = _rand(rng, 0.16, 0.27)
        fractal_dim = _rand(rng, 1.52, 1.90)
        fractal_bins = rng.randint(60, 110)
        ant_height = _rand(rng, 0.716, 0.730)
        soil_density = _rand(rng, 1.65, 1.95)
        dry_er = _rand(rng, 3.6, 5.0)
        dry_sigma = _rand(rng, 0.0015, 0.0035)
        moist_er = _rand(rng, 9.0, 13.5)
        moist_sigma = _rand(rng, 0.007, 0.018)
        wet_er = _rand(rng, 14.0, 20.0)
        wet_sigma = _rand(rng, 0.015, 0.032)
        clay_er = _rand(rng, 20.0, 28.0)
        clay_sigma = _rand(rng, 0.035, 0.060)

    oil_type = rng.choice(("gasoline", "diesel", "crude"))
    if oil_type == "gasoline":
        free_oil_er, free_oil_sigma = _rand(rng, 1.9, 2.1), _rand(rng, 0.0003, 0.0008)
    elif oil_type == "diesel":
        free_oil_er, free_oil_sigma = _rand(rng, 2.0, 2.3), _rand(rng, 0.0007, 0.0015)
    else:
        free_oil_er, free_oil_sigma = _rand(rng, 2.3, 2.8), _rand(rng, 0.0018, 0.0045)

    if profile == "sandbox_local_easy":
        # Keep free-oil in a tighter low-permittivity band for stronger visual separability.
        if oil_type == "gasoline":
            free_oil_er, free_oil_sigma = _rand(rng, 1.78, 1.98), _rand(rng, 0.00020, 0.00055)
        elif oil_type == "diesel":
            free_oil_er, free_oil_sigma = _rand(rng, 1.92, 2.12), _rand(rng, 0.00040, 0.00095)
        else:
            free_oil_er, free_oil_sigma = _rand(rng, 2.05, 2.30), _rand(rng, 0.00080, 0.00180)
        # Stronger EM contrast than host soil for competition-friendly visibility.
        oil_sat_er = _rand(rng, 2.0, 2.7)
        oil_sat_sigma = _rand(rng, 0.0002, 0.0008)
        surface_segments = 8
        if label == "oil":
            anomaly_lines, anomaly_meta = _positive_geometry_sandbox(
                rng=rng,
                scenario=chosen_scenario,
                tilt_max_deg=sandbox_tilt_max_deg,
            )
        else:
            anomaly_lines, anomaly_meta = _negative_geometry_sandbox(
                rng=rng,
                scenario=chosen_scenario,
                tilt_max_deg=sandbox_tilt_max_deg,
            )
        avoid_x = None
        if "oil_center_x" in anomaly_meta:
            avoid_x = float(anomaly_meta["oil_center_x"])
        elif "wet_center_x" in anomaly_meta:
            avoid_x = float(anomaly_meta["wet_center_x"])
        elif "pool_x0" in anomaly_meta and "pool_x1" in anomaly_meta:
            avoid_x = 0.5 * (float(anomaly_meta["pool_x0"]) + float(anomaly_meta["pool_x1"]))
        clutter_lines, clutter_meta = _sandbox_clutter(
            rng=rng,
            clutter_level=_clamp(sandbox_clutter_level, 0.0, 1.0),
            avoid_x=avoid_x,
        )
        anomaly_meta.update(clutter_meta)
    else:
        oil_sat_er = _rand(rng, 4.6, 6.8)
        oil_sat_sigma = _rand(rng, 0.0025, 0.0065)
        surface_segments = 14
        clutter_lines = _common_clutter(rng=rng)
        if label == "oil":
            anomaly_lines, anomaly_meta = _positive_geometry(
                rng=rng, scenario=chosen_scenario
            )
        else:
            anomaly_lines, anomaly_meta = _negative_geometry(
                rng=rng, scenario=chosen_scenario
            )

    if profile == "sandbox_local_easy":
        surface_lines = _surface_boxes(
            rng=rng,
            x_min=0.0,
            x_max=1.2,
            y_top=y_surface,
            segments=surface_segments,
            rough_min=-0.0025,
            rough_max=0.0015,
            min_thickness=0.003,
            max_thickness=0.010,
        )
    else:
        surface_lines = _surface_boxes(
            rng=rng, x_min=0.0, x_max=1.2, y_top=y_surface, segments=surface_segments
        )

    lines = [
        f"#title: Realistic shallow leakage scene ({label}, {chosen_scenario}, {profile})",
        f"#domain: 1.200 {air_top:.3f} 0.002",
        "#dx_dy_dz: 0.002 0.002 0.002",
        f"#time_window: {time_window:.3e}",
        f"#pml_cells: {pml_cells}",
        "",
        f"#material: {dry_er:.3f} {dry_sigma:.5f} 1.0 0.0 dry_topsoil",
        f"#material: {moist_er:.3f} {moist_sigma:.5f} 1.0 0.0 moist_sand",
        f"#material: {wet_er:.3f} {wet_sigma:.5f} 1.0 0.0 wet_sand",
        f"#material: {clay_er:.3f} {clay_sigma:.5f} 1.0 0.0 clay_lens",
        f"#material: {oil_sat_er:.3f} {oil_sat_sigma:.5f} 1.0 0.0 oil_saturated_sand",
        f"#material: {free_oil_er:.3f} {free_oil_sigma:.5f} 1.0 0.0 free_oil",
        "#material: 5.000 0.00050 1.0 0.0 pvc",
        "#material: 7.200 0.00300 1.0 0.0 gravel",
        "",
        f"#soil_peplinski: 0.52 0.14 {soil_density:.3f} 2.66 {water_low:.3f} {water_high:.3f} host_soil",
        (
            f"#fractal_box: 0.000 0.000 0.000 1.200 {y_surface:.3f} 0.002 "
            f"{fractal_dim:.2f} 1.0 1.0 1 {fractal_bins} host_soil z"
        ),
        f"#box: 0.000 {y_surface:.3f} 0.000 1.200 {air_top:.3f} 0.002 free_space",
        "",
        "## Surface roughness",
        *surface_lines,
        "",
        "## Main anomaly region",
        *anomaly_lines,
        "",
        "## Background clutter",
        *clutter_lines,
        "",
        f"#waveform: ricker 1 {freq_hz:.2e} txpulse",
        f"#hertzian_dipole: z 0.100 {ant_height:.3f} 0.001 txpulse",
        f"#rx: 0.125 {ant_height:.3f} 0.001",
        "#src_steps: 0.004 0.000 0.000",
        "#rx_steps: 0.004 0.000 0.000",
        "",
        "## Suggested command:",
        "## python -m gprMax <this_file>.in -n 220",
    ]

    meta: Dict[str, float | str | int] = {
        "label": label,
        "scenario": chosen_scenario,
        "frequency_hz": freq_hz,
        "seed": -1 if seed is None else seed,
        "profile": profile,
        "oil_type": oil_type,
        "fractal_dimension": round(fractal_dim, 3),
        "fractal_bins": fractal_bins,
        "water_low": round(water_low, 3),
        "water_high": round(water_high, 3),
        "antenna_height_y": round(ant_height, 3),
        "sandbox_tilt_max_deg": round(sandbox_tilt_max_deg, 3),
        "sandbox_clutter_level": round(sandbox_clutter_level, 3),
        "simplify_water": int(simplify_water),
    }
    meta.update(anomaly_meta)

    return "\n".join(lines) + "\n", meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate varied realistic gprMax .in")
    parser.add_argument("--output", required=True, type=Path, help="Output .in path")
    parser.add_argument(
        "--label",
        choices=["oil", "no_oil"],
        default="oil",
        help="Scene label (default: oil)",
    )
    parser.add_argument(
        "--scenario",
        default="auto",
        choices=("auto",) + ALL_SCENARIOS,
        help="Scenario type (default: auto)",
    )
    parser.add_argument(
        "--profile",
        default="realistic",
        choices=PROFILES,
        help="Simulation profile (default: realistic)",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=9.0e8,
        help="Ricker center frequency in Hz (default: 9e8)",
    )
    parser.add_argument(
        "--sandbox-tilt-max-deg",
        type=float,
        default=10.0,
        help="Max absolute tilt angle in degrees for sandbox anomalies (default: 10)",
    )
    parser.add_argument(
        "--sandbox-clutter-level",
        type=float,
        default=0.35,
        help="Sandbox clutter intensity in [0,1] for pipe/stone interferers (default: 0.35)",
    )
    parser.add_argument(
        "--simple-water",
        action="store_true",
        help="Use medium-low moisture band with reduced variability in sandbox profile",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=None,
        help="Optional JSON metadata output path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0.0 <= args.sandbox_clutter_level <= 1.0):
        raise ValueError("--sandbox-clutter-level must be in [0,1]")
    if args.sandbox_tilt_max_deg < 0.0:
        raise ValueError("--sandbox-tilt-max-deg must be >= 0")

    content, meta = build_scene(
        label=args.label,
        freq_hz=args.frequency,
        scenario=args.scenario,
        seed=args.seed,
        profile=args.profile,
        sandbox_tilt_max_deg=args.sandbox_tilt_max_deg,
        sandbox_clutter_level=args.sandbox_clutter_level,
        simplify_water=args.simple_water,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="ascii")
    print(f"Generated: {args.output.resolve()}")

    if args.meta_output is not None:
        args.meta_output.parent.mkdir(parents=True, exist_ok=True)
        args.meta_output.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="ascii")
        print(f"Metadata: {args.meta_output.resolve()}")


if __name__ == "__main__":
    main()
