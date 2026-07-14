# Security policy

Monolith is a desktop application with file, command, model, network,
peer-agent, and persistent-state capabilities. It is not a hardened multi-user
service or sandbox.

## Supported versions

Security fixes are considered only for the latest source on the default branch
and the latest published release. Older releases are unsupported.

| Version | Supported |
|---|---|
| Current `1.0.x` release/default branch | Best effort |
| Older releases | No |

## Reporting a vulnerability

Do **not** open a public issue for authentication bypass, arbitrary file or
command execution, credential disclosure, unsafe deserialization, remote
control, path escape, or another exploitable condition.

Use GitHub's **Report a vulnerability** private-reporting flow in this
repository's Security tab. If that option is unavailable, contact the
repository owner through the private contact method on their GitHub profile and
request a private channel without including exploit details in the first
message.

Include only what is necessary:

- affected commit or tag;
- Windows and Python versions;
- affected component and whether optional features were enabled;
- minimal reproduction steps;
- expected versus actual security boundary;
- impact and whether the issue is already being exploited; and
- sanitized logs or proof of concept.

Remove API keys, tokens, passwords, prompts, conversations, notes, model paths,
user names, room IDs, private endpoints, and unrelated personal data. Do not
upload an entire `%APPDATA%\Monolith` directory or database.

No fixed acknowledgement or remediation time is promised. The maintainer will
triage reports as capacity permits and may ask for coordinated disclosure while
a fix is prepared.

## CONNECT boundary

CONNECT defaults to `127.0.0.1`. Keep it there.

- Selecting an all-interface bind requires a non-empty
  `MONOLITH_AGENT_TOKEN`; startup is blocked without one.
- Authenticated CONNECT routes, including `/events`, require that token.
- A token does not provide TLS, per-user roles, tenant isolation, rate-limited
  internet hosting, or secure reverse-proxy configuration.
- Remote or internet exposure is unsupported for this release.
- Anyone with access to an authorized CONNECT client may receive prompts,
  response tokens, state, traces, or tool/runtime events depending on the
  route and active features.

Use operating-system firewall rules and process isolation in addition to the
application token. Do not place CONNECT behind a public tunnel.

## Other high-risk surfaces

- File and command tools run with the operating-system permissions of the
  Monolith process. Governance levels are not an OS sandbox.
- API model backends receive prompt/context data configured by the active turn.
- Web search and grounding can send queries to Tavily.
- Matrix transmits messages and metadata to the configured homeserver.
- External peers and webhooks send data to user-supplied URLs.
- Model files and native ML dependencies are third-party code/data with their
  own trust and license boundaries.
- Configuration, chat archives, traces, and memory databases may contain
  secrets or private content in plaintext.
- Experimental self-maintenance and identity/memory mutation paths should not
  be enabled unattended.

## Secrets

Never commit or attach:

- `%APPDATA%\Monolith\config`;
- `config.yaml`, `tavily.json`, Matrix state, or addon credential files;
- `.env` files or launcher scripts containing credentials;
- `MONOLITH_AGENT_TOKEN`, `MONOLITH_PEER_TOKEN`, `TAVILY_API_KEY`, provider
  keys, Matrix passwords/tokens, or webhook secrets; or
- logs/screenshots containing the values above.

The general model API key entered in the UI is currently persisted locally in
plaintext configuration. Use a restricted key, protect the Windows account,
and rotate any credential that may have entered Git history or a shared file.

## Scope notes

Reports about exploitable boundary violations are welcome. General model
hallucination, prompt quality, unsupported remote deployment, denial of service
from intentionally oversized local models, and behavior caused solely by an
untrusted model/provider are not automatically security vulnerabilities, though
they may still be valid bugs.

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for state and secret
handling and [`docs/agent-guides/CONNECT.md`](docs/agent-guides/CONNECT.md) for
the local CONNECT protocol.
