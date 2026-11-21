# 017: Log corrections — duplicate number and stale doc lines
Date: 2026-11-21   Follows: [016](016-2026-11-21-step3h-full-network.md)

**What:** This entry corrects numbering and stale text in the log. No result changes.
**Corrections:**
- Two entries have the number 009: [009 confident 7s](009-2026-11-20-step3-confident-7s.md) and [009 removal test](009-2026-11-20-step3-removal-test.md). Both files stay unchanged, because entries are append-only. Read the confident-7s entry as 009a and the removal-test entry as 009b.
- Two experiment docs had the number 03. The confident-7s doc is now [experiments/04-confident-7s.md](../docs/experiments/04-confident-7s.md). Old links in entries to `03-confident-7s.md` refer to this file.
- `findings.md` said "one seed" for the removal test. The removal test now has 3 seeds on test image 4433.
- The question and setup of `experiments/03-removal-test.md` described only test image 0 at k = 100. They now describe all runs.
- `open_questions.md` listed two done items (a test image with a larger loss, a misclassified test image). They are removed.
