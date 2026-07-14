---
name: web
description: Fetch URL content (text-extracted or raw); blocks private/loopback IPs.
---

{"tool":"web","verb":"text","url":"https://example.com/"}
Verbs: text (default, HTML stripped to readable text) / fetch (raw body bytes decoded)
Optional: max_chars (200-50000, default 4000), timeout (default 15s)
Only http/https; blocks private/loopback/link-local/multicast/unspecified IPs.
