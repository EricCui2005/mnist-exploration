# 002: Project restart with incremental steps
Date: 2026-11-20   Follows: [001](001-2026-11-20-prototype-pipeline.md)

**What:** Eric removed all prototype code, data, and logs from the project. The files are in `~/.Trash/omri-old-20261120-161103`.
**Why:** The prototype grew too fast. Eric could not explain each design choice in it.
**Decisions:**
- Restart from an empty project — understanding has priority over speed.
- Use the `incremental-experiments` skill — one question per step, a proposal before each run.
- Keep this log — record each step and its reason.

**Next:** Step 1 examines if the stored keys and errors rebuild the trained weights.
