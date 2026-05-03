# gprMax Simulation Quick Start

## A. Single scene -> single B-scan (full pipeline)

```bash
python scripts/run_gprmax_bscan.py --input sim_inputs/oil_leak_realistic.in --traces 220 --output-image outputs/oil_leak_realistic.png --resize 256 256
```

With per-trace zero-time correction + background removal:

```bash
python scripts/run_gprmax_bscan.py --input sim_inputs/oil_leak_realistic.in --traces 220 --output-image outputs/oil_leak_realistic_zt.png --resize 256 256 --zero-time-correct --zero-time-search-ratio 0.15 --zero-time-smooth 9
```

GPU mode:

```bash
python scripts/run_gprmax_bscan.py --input sim_inputs/oil_leak_realistic.in --traces 220 --gpu --output-image outputs/oil_leak_realistic_gpu.png --resize 256 256
```

## B. Generate one randomized scene (.in only)

Oil sample:

```bash
python scripts/generate_gprmax_in.py --output sim_inputs/random_oil_0001.in --label oil --scenario auto --seed 101
```

No-oil hard negative sample:

```bash
python scripts/generate_gprmax_in.py --output sim_inputs/random_no_oil_0001.in --label no_oil --scenario auto --seed 102
```

Available scenarios:

- Oil: `plume_diffuse`, `pool_capillary`, `layered_sheet`, `pocket_chain`
- No-oil: `wet_lens`, `clay_lens`, `buried_pipe`, `mixed_clutter`

## C. Full automation for many different B-scans

Small smoke test (recommended first):

```bash
python scripts/build_bscan_dataset.py --count 6 --traces 48 --output-root dataset_auto_smoke --keep-in-files
```

Formal dataset generation:

```bash
python scripts/build_bscan_dataset.py --count 200 --traces 220 --oil-ratio 0.5 --frequencies 6.0e8,8.0e8,9.0e8,1.0e9 --output-root dataset_auto --keep-in-files
```

## D. Sandbox local leak (near-surface, clearer feature) test

Oil-only sandbox smoke test:

```bash
python scripts/build_bscan_dataset.py --count 12 --traces 48 --oil-ratio 1.0 --profile sandbox_local_easy --frequencies 9.0e8,1.0e9 --output-root dataset_sandbox_local_oil --keep-in-files
```

Mixed sandbox test (positive + hard negatives):

```bash
python scripts/build_bscan_dataset.py --count 30 --traces 64 --oil-ratio 0.6 --profile sandbox_local_easy --frequencies 9.0e8,1.0e9 --output-root dataset_sandbox_local_mix --keep-in-files
```

Competition-friendly sandbox (obvious contrast + wider depth spread + medium-low moisture):

```bash
python scripts/build_bscan_dataset.py --count 1000 --start-index 37 --traces 64 --oil-ratio 0.5 --profile sandbox_local_easy --frequencies 8.0e8,9.0e8,1.0e9 --sandbox-tilt-max-deg 16 --sandbox-clutter-level 0.35 --simple-water --output-root dataset_sandbox_local_1k_comp_v5_obvious --python-exe C:\\Users\\12161\\anaconda3\\envs\\gprMax\\python.exe --resize 256 256 --keep-in-files --mute-top-ratio 0.08 --mute-fade-ratio 0.03 --gpu
```

GPU mode:

```bash
python scripts/build_bscan_dataset.py --count 200 --traces 220 --oil-ratio 0.5 --frequencies 6.0e8,8.0e8,9.0e8,1.0e9 --output-root dataset_auto_gpu --gpu --keep-in-files
```

## E. Train MobileNetV3-Small (oil/no_oil)

Install dependencies first:

```bash
pip install torch torchvision pillow numpy
```

Train with default 70/15/15 split:

```bash
python scripts/train_mobilenetv3_small.py --data-root dataset_sandbox_local_1k_comp_v2/images --output-dir runs/mobilenetv3_small_1k --epochs 40 --batch-size 32 --image-size 224 --lr 3e-4 --amp
```

Train with stronger mirror + rotation augmentation:

```bash
python scripts/train_mobilenetv3_small.py --data-root dataset_sandbox_local_1k_comp_v2/images --output-dir runs/mobilenetv3_small_1k_aug --epochs 40 --batch-size 32 --image-size 224 --lr 3e-4 --hflip-prob 0.5 --rotate-deg 10 --amp
```

Key outputs:

- `runs/mobilenetv3_small_1k/checkpoints/best.pt`
- `runs/mobilenetv3_small_1k/checkpoints/last.pt`
- `runs/mobilenetv3_small_1k/config.json`
- `runs/mobilenetv3_small_1k/history.json`
- `runs/mobilenetv3_small_1k/test_metrics.json`

Generated structure:

- `dataset_auto/images/oil/*.png`
- `dataset_auto/images/no_oil/*.png`
- `dataset_auto/inputs/*.in` (if `--keep-in-files`)
- `dataset_auto/out/*_merged.out` (if `--keep-merged-out`)
- `dataset_auto/metadata.csv`

## F. Infer X-axis Leak Heatmap for One B-scan

Use a trained checkpoint (`best.pt`) to estimate oil-leak probability along the X axis.

```bash
python scripts/infer_leak_heatmap.py --image <path_to_one_bscan_png> --checkpoint <path_to_best.pt> --output-prefix outputs/demo_leakprob --stride-px 4 --smooth-k 9 --amp
```

Outputs:

- `*_overlay.png`: probability overlay on original B-scan
- `*_heatbar.png`: pure top heat bar
- `*_combined.png`: heat bar + overlay combined view
- `*_xprob.csv`: x-position probability table
- `*_summary.json`: summary with peak positions

## G. Stepped-Frequency Oil-Leak Scene (720~2500 MHz)

Generate fixed-parameter sweep input files (your specified hertzian dipole setup and 3-layer oil zone):

```bash
python scripts/generate_sfcw_leak_sweep.py --output-dir sim_inputs/sfcw_oil_720_2500 --f-start-mhz 720 --f-stop-mhz 2500 --f-step-mhz 20 --waveform contsine --traces 96 --tx-rx-spacing 0.1 --tx-x0 0.06 --trace-step 0.003
```

Then run one frequency:

```bash
python -m gprMax sim_inputs/sfcw_oil_720_2500/sfcw_oil_0720MHz.in -n 96
```
