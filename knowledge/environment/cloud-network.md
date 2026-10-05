---
type: Environment Fact
title: Cloud session network access
description: Which hosts the cloud environment reaches, which need allowlisting, and the OpenML host quirk.
tags: [environment, network, cloud, openml]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-05T22:05:00Z }
stale_after: 2027-01-05T00:00:00Z
sources:
  - id: proxy
    resource: process:agent-proxy status endpoint ($HTTPS_PROXY/__agentproxy/status)
    title: Agent proxy status, 2026-10-05
  - id: settings
    resource: claude.ai cloud environment settings, Network access
    title: Environment "Default", Custom network access
    author: human:ekthesage
---

# Facts (2026-10-05)

* Network access is Custom with allowed domains `www.openml.org` and
  `api.openml.org`, added by the user; a running session picks up the
  change without restarting.[^settings]
* Package registries bypass the proxy: PyPI (`pypi.org`,
  `files.pythonhosted.org`), crates.io, npm, Go proxy. `pip install
  statsmodels` and `pip download bayesblend` work.[^proxy]
* `github.com` is reachable for `git clone` of public repositories.
* Bare `openml.org` is refused (CONNECT 403), though OpenML's metadata
  links to it; the same paths work on `www.openml.org` (see
  [freMTPL2](/datasets/fremtpl2.md)).
* pkgdown's site build (`cargo xtask r`, last step) fails with CONNECT 403
  on a host it fetches; the R tests before it are unaffected.

[^settings]: Environment settings
[^proxy]: Agent proxy status
