# 019: The MNIST work moves to `mnist_exploration/`
Date: 2026-11-21   Follows: [018](018-2026-11-21-step3i-five-misclassified.md)

**What:** All scripts, logs, figures, data and this project log move from `omri/` to `omri/mnist_exploration/`.
**Why:** The `omri/` folder will hold several experiment folders. Each folder keeps its own scripts and project log.
**Decisions:**
- The `uv` environment (`pyproject.toml`, `uv.lock`, `.venv`) stays in `omri/` — all experiment folders share it. `uv run` finds it from a subfolder.
- `.claude/` (the skills) stays in `omri/` — the skills apply to all experiment folders.
- `data/` moves with the scripts — the scripts use the relative path `data`.

**Result:** `step2_attention.py --error` runs from `mnist_exploration/` and gives the same output. Relative links in the log do not change.
