# V1 labeling pass — surface_note / curiosity triage (read-only; ledgers + kill-candidates in %APPDATA%/Monolith/logs/)

Run: `python -m tools.labeling.cli` (surface_note) or `--curiosity` (canonicals partition, never pooled). Blind-first; h/i/d; 20/batch stratified by species and date terciles, seed stored per row.

**Graduation rule:** the first `hit` label = an action external to this apparatus; it unlocks drafting V1.1 only — nothing in V1 acts on labels automatically.

**Kill rule:** 40 consecutive duds, or two missed 14-day labeling windows (N=14 — measures the corpus's pull, not store-calendar compression) → archive both ledgers + write a verdict note; apparatus retired.

Curiosity duds land in `curiosity_kill_candidates.txt` for E's review — `kill_pull` is never called from here.

Verdict-note convention: any non-dud label recorded while the dud streak is within 5 of the 40-dud threshold gets an asterisk in the verdict note — the streak display is the one scoreboard element that can corrupt what it measures, and this names the pressure without building against it.

