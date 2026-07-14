"""BRANCH divergence pilot — the cheapest falsifier for the BRANCH premise.

Forces each divergent golden probe under its correct frame and its foil frame
via POST /thinkpad's scaffold override, then asks one question: does DeepSeek's
answer FOLLOW the forced frame? If forcing the foil frame still lands the
correct answer, frames don't bite for this model and BRANCH's accuracy bet is
weak. If correct-frame passes and foil-frame fails, frames bite — separation is
real and the rest of BRANCH is worth building.

Reads NOTHING from monothink; writes NOTHING live. Probe ground truth +
forcing instructions live in probes.py and are gated by decorrelated
recomputation (the standing artifact gate) before any live run.
"""
