---
name: session
description: Query current chat session via canonical_log (state, recent events, search).
---

{"tool":"session","verb":"recent","limit":10}
Verbs: state / recent / search
Optional: limit (recent: 1-100 default 10; search: 1-100 default 20), pattern (verb=search, substring match)
