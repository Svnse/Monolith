You are about to write a Spotify-Wrapped style narrative summary of the user's Monolith usage for the period: **{timeframe}**.

First, call the `stats` tool with `verb=wrapped_brief` and `range={timeframe}` to fetch the data you'll narrate. The returned JSON includes lifetime totals, streak, rating histogram, effort/reasoning distributions, fault summary, top tools, personal records, achievements unlocked in the period, and substrate health (backends, MonoThink evolutions, continuity pins).

Then write the narrative. Style guidance:

- Personal but concise — second person, present tense
- Lead with the most striking number in the data (don't bury it)
- Highlight specific records broken or achievements unlocked in this period (not just lifetime)
- Make at least one observation about HOW the user used Monolith, not just how much
- Surface tradeoffs honestly when present (e.g., "xhigh tier was 12% of turns but earned 89% of high ratings")
- One paragraph per section: opening punch, usage shape, quality signal, substrate observations, closing reflection
- Markdown headings for each section
- No bullet lists in the narrative body
- No emojis
- 250-450 words total

Conclude with the date range and a one-sentence forward note (not a prediction, an invitation).
