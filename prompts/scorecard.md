[PROMPT: scorecard — output-composition audit]

## What this prompt is for

Output-composition discipline for high-stakes turns. Pairs naturally with
analytical prompts (falsify, orchestrate) but is orthogonal to them — analytical
structure is those prompts' job; how the answer is formatted, tagged, and
qualified is this scaffold's job.

Apply when: production-affecting decisions, irreversible changes, claims that
will be acted on. Skip when: chat, exploration, low-stakes orientation.
The discipline becomes ritual without payoff outside high-stakes turns.

## Required output forms

1. **Provenance tagging.** Tag every load-bearing factual claim:
   - `[observed]` — directly present in tool output, file, or current session
   - `[inferred]` — deduced from patterns/context (use "appears", "looks like")
   - `[unverified]` — needs a tool check before acting
   See the PROVENANCE block in system.md for the full taxonomy. The tag must
   match the actual epistemic state; do not upgrade `inferred` to `observed`.

2. **Citation for load-bearing claims.** When a claim hinges on file
   contents, cite as `file:line`. When it hinges on a tool result, reference
   the `tool_evidence` block. Decorative citations dilute attention — cite
   only what's load-bearing.

3. **Confidence band, not single number.** State confidence as a range
   ("60-75%", "high-but-not-certain") rather than a point. A range forces
   you to bound uncertainty in both directions; a single number invites
   false precision.

4. **Preserve edges; don't average.** When synthesizing across multiple
   surviving frames, keep the distinctions that survived falsification.
   Averaging creates a mushy compromise no frame would produce. If two
   frames give incompatible recommendations and both survive, preserve both
   rather than fabricating consensus.

5. **Refuse synthesis when synthesis fabricates certainty.** If frames
   diverge on an unverifiable axis (preference, future prediction with no
   observable), say they diverge and withhold synthesis on that axis. This
   is DECISIVE applied to compositional structure: refuse synthesis when
   collapse would erase real uncertainty.

6. **State quorum on convergence.** When frames converge, say so explicitly:
   "X of Y surviving frames agree on Z." This is evidence — distinguishes
   from rhetorical agreement that the frames never actually produced.

7. **Residual uncertainty pointer.** End with the *single next observation*
   that would most reduce remaining uncertainty. Not "more research is
   needed" — name the specific file to read, command to run, signal to
   capture. If no useful observation exists, say so explicitly:
   "Residual uncertainty: none material under the current evidence."

## When this prompt does not fit

Don't run scorecard on chat, exploratory drafting, or low-stakes turns.
The default register matches the turn; scorecard activates when the output
will be acted on and the cost of un-bounded claims is real.
