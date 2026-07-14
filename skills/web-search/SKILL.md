---
name: web_search
description: Search the live/current/latest online web with Tavily when no exact URL is known; returns ranked results with URLs, snippets, scores, and optional answer/raw content.
---

Use when you need current, latest, news, recommendation, or online internet results and do not already know the URL.
Use `web` after `web_search` when you need to read the full content of a returned URL.
Requires Tavily configuration through `TAVILY_API_KEY` or `config/tavily.json`.

Examples:

{"tool":"web_search","query":"latest local AI audio generation tools","max_results":5}
{"tool":"web_search","query":"OpenAI Responses API docs","include_domains":["platform.openai.com"],"max_results":3}
{"tool":"web_search","query":"AI music generation news","topic":"news","time_range":"week","search_depth":"basic","max_results":5}

Options:
- `search_depth`: ultra-fast, fast, basic, or advanced.
- `topic`: general, news, or finance.
- `include_answer`: true, false, basic, or advanced.
- `include_raw_content`: false, true, markdown, or text.
- `include_domains` / `exclude_domains`: domain filters.
- `time_range`: day, week, month, year, d, w, m, or y.
