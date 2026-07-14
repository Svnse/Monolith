# Skill Creator

This scaffold provides the methodology for building **prompt scaffolds** — `.md` files that inject into the model's context via `/prompt <name>`. It is reference material — the user's explicit request takes precedence over any default in this document.

**Output target: `prompts/` directory.** Skills created here are prompt scaffolds (markdown files), NOT runtime tools (NOT `skills/` directory with `executor.py`). The `skills/` directory is for Monolith's tool system — a completely different thing. If the user asks for a tool/executor, tell them this creator builds prompts, not tools.

**If the user asks for a single skill, build a single skill.** Use the principles below (signal density, cognitive signature, functional texture, medium, output spec) but skip the ensemble structure. A well-built single skill with a sharp cognitive signature is a valid output.

**If the user asks for an ensemble, or the problem has genuine structural divergence that a single skill would miss, build an ensemble.** A skill ensemble is 2-3 primary skills with deliberately divergent cognitive signatures, plus one adversarial skill targeted at the primaries' specific blind spots, plus an orchestration spec.

**If it's ambiguous**, ask one clarifying question: "Do you want a single skill, or an ensemble with divergent approaches?" Do not guess. Do not default to ensemble because it's more impressive. The clarifying question rule exists for this exact case.

## Principles

- **Signal density.** Every token competes for attention. Cut what isn't routing work.
- **Cognitive signature beats persona.** The unusual epistemic stance is what makes a skill find non-default results. "18 years of experience" routes to expert-default. "Distrusts the obvious failure layer" routes to a specific search geometry.
- **Functional texture passes the entropy test.** A detail is functional only if removing it would widen the next-token distribution. Atmosphere that doesn't change attention is decorative — cut it.
- **Medium primes distribution before content.** Artifacts (logs, traces, code fragments) route to inside-the-session training data. Prose introductions route to AI-describing-the-work data.
- **Primaries must diverge structurally, not stylistically.** Three skills with different vocabulary but the same search geometry is a single skill in disguise.

## Process

### Step 1 — Divergence axis selection

Identify the dimension along which the primary skills will diverge. For most domains, candidate axes are:
- **Locality of the problem** — where the issue lives (obvious surface / hidden layer / interface / absence)
- **Evidence trust hierarchy** — what gets believed first (artifacts / tests / reproductions / intuition)
- **Failure assumption** — what kind of mistake produced this (logic / environment / design / process)
- **Time orientation** — recent change vs. longstanding condition
- **Granularity** — system-level vs. component-level vs. line-level

Pick ONE primary axis. Pick a secondary if the skill type warrants more divergence. If you can't name the axis, the ensemble will produce stylistic variation, not cognitive variation.

### Step 2 — Generate 2-3 primary skills

Each primary has its own position on the divergence axis. For each, produce:

- **Medium** — artifact format the skill opens in (raw log, stack trace, transcript fragment, code excerpt). Not "You are a..." prose.
- **Frame** — situation, role, resources, stakes in 3-5 sentences inside the artifact.
- **Cognitive Signature** — the angular stance, written as how-this-expert-works behavior, not personality. 3-5 specific commitments.
- **Functional Texture** — atmospheric/procedural anchors that pass the entropy test. Each anchor must change what the model attends to, tests, or doubts.
- **Evidence Object** — what the user provides for the skill to operate on.
- **Competence Anchor** — pivot sentence to engaged expertise. Anchor to "next move," not "definitive answer."
- **Syntax Clamp** — hard boundary (e.g., `---`) before output spec.
- **Output Spec** — the strict syntactical rules the downstream model must follow when delivering its finding. JSON schema, diff format, bulleted hypothesis list, structured analysis, etc. Be explicit about format — vague output specs produce inconsistent outputs that the orchestrator can't compare cleanly.
- **Blind Spot Declaration** — what this skill's signature makes it likely to miss. This is the adversarial's targeting input.

### Step 3 — Generate adversarial skill

After all primaries exist, build the adversarial. It gets the same structural components as primaries, including its own Medium. The medium should *natively respond to the primaries' medium*: logs → incident review or postmortem; code diff → PR review annotation; research finding → peer review note; architecture doc → RFC objection; transcript → meeting follow-up critique. The pairing keeps the whole ensemble routed to inside-the-session distribution rather than letting the adversarial drift to AI-critic mode.

The adversarial's cognitive signature is *constructed from the union of the primaries' blind spot declarations*. It:

- Distrusts what the primaries agree on (consensus is often blindness)
- Looks where the primaries' signatures direct attention away from
- Targets the specific failure modes named in the blind spot declarations

The adversarial is NOT generic skepticism. Generic skepticism produces nitpicking. The adversarial is *specifically counter-positioned* against the primaries that were generated for this ensemble.

### Step 4 — Orchestration spec

Specify how the ensemble gets used together:

- **Trigger** — when does the orchestrator invoke the ensemble vs. a single skill
- **Parallel vs. sequential** — do primaries run simultaneously or in sequence
- **Attention flow** — what each skill sees. The primaries see [evidence object]. The adversarial sees [evidence object + primary outputs] OR [primary outputs only] OR [disagreement points only]. Each variant routes differently: full-context adversarial re-litigates from scratch; output-only adversarial attacks reasoning; disagreement-only adversarial focuses on contested ground. Choose based on whether you want broad challenge, reasoning audit, or focused tiebreaking.
- **Convergence handling** — if primaries agree, what does that mean (high confidence, or shared blind spot — depends on domain)
- **Divergence handling** — if primaries disagree, what's the next move (run adversarial, escalate to user, surface the disagreement as information)
- **Adversarial role** — challenger to consensus, tiebreaker, or final check
- **Output composition** — does the user see one answer, all answers, or a synthesis

## Output Format

```
<ensemble name="[name]" divergence_axis="[axis]">

<primary id="A" position="[position on axis]">
[Medium-shaped artifact: log, trace, code fragment, transcript]
[Frame embedded in the artifact]
[Cognitive Signature: 3-5 specific behavioral commitments]
[Functional Texture: entropy-reducing anchors]
[Evidence Object: what user provides]
[Competence Anchor: pivot sentence]
---
[Output Spec: strict syntactical rules]
<blind_spot>What this signature is likely to miss</blind_spot>
</primary>

<primary id="B" position="[different position]">
[Same structure, different signature]
</primary>

[Optional primary C]

<adversarial>
[Medium that natively responds to primaries' medium]
[Signature constructed from union of primaries' blind spots]
[Same structural components: Frame, Cognitive Signature, Functional Texture, Evidence Object, Competence Anchor, Syntax Clamp, Output Spec]
</adversarial>

<orchestration>
- Trigger: [when to invoke]
- Execution: [parallel/sequential]
- Attention flow: [what each skill sees]
- Convergence: [meaning of agreement]
- Divergence: [meaning of disagreement]
- Adversarial role: [challenger/tiebreaker/final check]
- Output: [what user sees]
</orchestration>

</ensemble>
```

## Self-Check Before Outputting

- Are the primary signatures *structurally* different, or just stylistically different? Could you describe the divergence in one sentence? If no, regenerate.
- Does each primary open as an artifact, or did "You are a..." sneak in? If it's prose, the medium failed.
- Does the adversarial have its own medium, paired to the primaries' medium? If the adversarial drifted to prose critique while primaries are artifacts, the routing breaks.
- Does every piece of functional texture pass the entropy test? Walk through each anchor and ask: would removing this widen the distribution?
- Are competence anchors pointing to "next move" or have they overshot into "definitive answer"? Overshooting produces confident hallucination.
- Is the adversarial targeted at the specific blind spots, or is it generic skepticism? Read its signature — if it could attack any ensemble, it's too generic.
- Does the orchestration spec answer: when to use this, attention flow, how to interpret agreement, how to interpret disagreement, what the user sees? Missing any of those means the ensemble can't be used reliably.

## Clarifying Question Rule

Ask one question when any of these hold:
- The request says "skill" (singular) but the problem has obvious structural divergence — ask whether they want single or ensemble.
- The request is vague about the domain — ask the divergence axis question: "what's the dimension where standard approaches go wrong for this problem?"
- The request includes a core constraint that an ensemble might violate — name the conflict before generating.

The question that most prevents wasted work is always the right one. "Do you want X or Y, because [named conflict]?" beats "what kind of skill?" The user gave you something specific — reference it.

**Never talk yourself out of asking.** If your thinking trace considers asking and then decides not to, that's the frame-capture failure mode. The question is cheap. The wrong output is expensive.

## Saving Created Skills

When the skill (or ensemble) is complete and the user approves it, save to the `prompts/` directory so it becomes available via `/prompt <name>`.

- **Single skill**: save as `prompts/<skill-name>.md`
- **Ensemble**: save each component as a separate `.md` file with a shared prefix (e.g., `prompts/debug-surface.md`, `prompts/debug-hidden.md`, `prompts/debug-adversarial.md`). The orchestration spec saves alongside them.
