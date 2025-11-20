# Repository Guidelines

## Project Structure & Module Organization
- Core entrypoints live at the repo root: `ace_zero.py` (end‑to‑end reconstruction), `train_ace.py` (ACE training), `register_mapping.py` (camera registration), plus utilities such as `dataset.py`, `dataset_io.py`, and `ace_*.py`.
- Dataset helpers are in `datasets/`, benchmarking and Nerfstudio integration in `benchmarks/`, point‑cloud diffusion priors in `diffusion/`, C++ DSAC* bindings in `dsacstar/`, and dataset/experiment scripts in `scripts/`. Sample data layout is illustrated by `TexturePoorSfM_dataset/`.

## Build, Test, and Development Commands
- This repository is maintained as a Pixi-based environment. When running commands (including tests), prefer `pixi run <command>` over manual environment activation.
- Create the environment: either `pixi install` (recommended) or `conda env create -f environment.yml` then `conda activate ace0`.
- Build DSAC* bindings (required before running ACE0): `cd dsacstar && pixi run python setup.py install` (or `pixi run ext`).
- Run a basic reconstruction: `pixi run python ace_zero.py "/path/to/images/*.jpg" result_folder`.
- Evaluate poses via Nerfstudio benchmark: `pixi run python -m benchmarks.benchmark_poses --pose_file poses_final.txt --images_glob_pattern "/path/to/images/*.jpg" --output_dir benchmark_out`.

## Coding Style & Naming Conventions
- Python throughout, 4‑space indentation, and PEP8‑style naming: `snake_case` for functions/variables/modules, `CamelCase` for classes, `UPPER_CASE` for constants.
- Mirror local patterns: prefer small, single‑purpose functions, explicit imports, and `_logger = logging.getLogger(__name__)` for logging.
- No global reformatting or style tool changes; keep diffs minimal and confined to the feature or bug you are touching.

## Testing Guidelines
- There is no dedicated unit‑test suite; instead, use small ACE0 runs and the benchmarking tools as regression checks, always invoked via `pixi run`.
- For changes affecting reconstruction, run `pixi run python ace_zero.py ...` on at least one small scene and verify that `poses_final.txt` and optional `pc_final.ply` are produced without errors.
- For changes affecting pose quality or priors, run a short benchmark via `pixi run python -m benchmarks.benchmark_poses ...` and summarize PSNR or related metrics in your PR.

## Commit & Pull Request Guidelines
- Follow existing history: short, imperative or descriptive messages (e.g., “Add RGB‑D reconstruction prior”, “Fix Docker compose image path”), one logical change per commit.
- For pull requests, include: a concise summary, affected scripts/modules, dataset(s) used for validation, key command lines, hardware details if relevant, and any expected change in reconstruction or benchmark quality.
- Update `README.md` or script docstrings when behavior, flags, or expected outputs change.

## Agent-Specific Instructions
- When editing this repository programmatically, respect these guidelines, avoid broad refactors, and prefer incremental, well‑scoped changes.
- When running or debugging code via shell commands, always execute inside the Pixi environment using `pixi run <command>` (for example, `pixi run python ace_zero.py ...`), rather than calling `python` or other tools directly.
- Do not add new heavy dependencies or commit dataset artifacts without prior discussion in an issue or PR description.
