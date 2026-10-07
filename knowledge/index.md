---
okf_version: "0.2"
---

# prospicio knowledge bundle

Knowledge collected while building prospicio, in the
[Open Knowledge Format v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md):
facts about data, reference implementations and the build environment,
and findings that explain why the code is the way it is. Design decisions
stay in [docs/design](../docs/design/); concepts here link to them.

* [datasets](datasets/index.md) - Data used for parity and validation: where it comes from, how it is prepared, what it contains.
* [references](references/index.md) - External reference implementations the parity tests compare against, and their quirks.
* [findings](findings/index.md) - Technical results found while building: biases, edge cases, numerical facts.
* [environment](environment/index.md) - Facts about the build and agent environment: network access, tooling, local-check noise.

# Conventions

* `generated.by` is `claude-code/cloud-session` for content the cloud session writes, `human:<id>` for people.
* A fact confirmed by a parity or unit test that CI runs is `verified` by `process:ci`, at the time of the CI run that merged it.
* Environment facts carry a `stale_after`: they describe a configuration that can change.
* Every concept has `type`, `title`, `description` and `tags`; `validation/scripts/check_okf.py` checks conformance in CI.
