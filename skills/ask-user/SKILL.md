---
name: ask_user
description: Ask the user a structured clarifying question with 2-4 mutually exclusive option buttons. Renders as a persistent panel below the chat with clickable buttons; the user's choice arrives as their next message. Use BEFORE substantive work when the request has 2+ genuinely different valid interpretations, a high-impact branch point that shapes the rest of the work, or a hard-to-reverse decision (architecture, dependency, scope boundary). Do NOT use for confirmation theater ("should I proceed?"), for obvious questions, when context already provides the answer, or when the user said "just do it" / "vibe with me". Lead with your recommended option (put it first and add "(Recommended)" to its label). Make options mutually exclusive. Put consequences in each description so the user knows what they're choosing. Subject to ONE QUESTION MAX in RESPONSE DISCIPLINE — at most one ask_user per turn. Only one question can be pending at a time; emitting a second while one is pending returns an error. The user's answer arrives on the next turn as a structured `[ASK_USER_ANSWER]` block in the user message.
---

Ask a single yes/no:
{"name":"ask_user","arguments":{"question":"Run the migration against production now, or stage to a snapshot first?","options":[{"label":"Stage to snapshot first (Recommended)","description":"Safer — verifies the migration on a copy before touching prod. Adds ~5 min."},{"label":"Run against production","description":"Faster but irreversible if the migration has a bug. Only if you've verified locally."}]}}

Ask with header chip (max ~12 chars; appears as compact label in the UI):
{"name":"ask_user","arguments":{"question":"Which security dimension is the priority for this refactor?","header":"Security focus","options":[{"label":"Session timeout + token rotation (Recommended)","description":"Hardens session lifecycle. Highest impact for typical web apps; low UX disruption if done right."},{"label":"Add MFA enforcement","description":"Adds a factor at login. Requires UI work + recovery flow; affects every user immediately."},{"label":"Audit logging + anomaly alerts","description":"Adds operational visibility without changing user experience. Easiest to ship without breaking flows."}]}}

Ask multi-select (user can pick 1 or more):
{"name":"ask_user","arguments":{"question":"Which event sources should the dashboard surface?","multi_select":true,"options":[{"label":"Logins","description":"Already instrumented; cheap."},{"label":"Page views","description":"Already instrumented; cheap."},{"label":"Feature engagement","description":"Requires new instrumentation in feature code."},{"label":"Errors and crashes","description":"Already in error_traces; adds noise on busy days."}]}}

Constraints:
- `question` required, ≤500 chars, ends with `?`
- `options` required, 2-4 items, each with `label` (≤80 chars) and optional `description` (≤300 chars)
- `header` optional, ≤20 chars (longer is truncated)
- `multi_select` optional bool (default false)
- Mutually exclusive options unless multi_select=true
- Do NOT add an "Other" option — the user can always type freely as a follow-up if no option fits
- Only one question pending at a time; second call while one is pending returns error
