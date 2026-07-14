# Summary

Describe the user-visible problem and the smallest mechanism changed.

## Runtime path

Describe the affected `trigger -> producer -> consumer -> visible or next-turn
effect` path.

## Verification

- Exact commands/tests:
- Result, including pre-existing failures:
- Manual UI/runtime check:
- Isolated `MONOLITH_ROOT` used: yes/no/not applicable

## Contract and risk checklist

- [ ] The change is scoped; unrelated user work and formatting are preserved.
- [ ] Tests cover the changed behavior or the reason they cannot is stated.
- [ ] README, known issues, changelog, and public docs are updated as needed.
- [ ] New/changed feature flags document default, writes, network use, and rollback.
- [ ] Persistent-state/schema changes include migration and backup implications.
- [ ] File, command, model, network, authentication, and privacy effects are explained.
- [ ] New dependencies are placed in the correct group and their license boundary is reviewed.
- [ ] Copied/adapted code, prompts, workflows, screenshots, or assets have provenance.
- [ ] Screenshots/logs contain no keys, prompts, private paths, endpoints, identities, or personal data.
- [ ] Work intentionally deferred is listed below.

## Deferred work

List relevant work this pull request intentionally does not complete.
