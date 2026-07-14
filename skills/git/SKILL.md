---
name: git
description: Query git repository state (status, diff, log, branch).
---

{"tool":"git","verb":"status"}
Verbs: status / diff / log / branch
Optional: cwd (default: process cwd), limit (verb=log, 1-100, default 10), full (verb=diff, true=full diff, false=--stat)
