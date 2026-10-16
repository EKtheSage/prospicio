---
type: Environment Fact
title: Signing the Excel XLL
description: How the XLL is Authenticode-signed with osslsigncode on Linux, what was checked with a self-signed certificate, and which certificates fit a CI secret.
tags: [environment, excel, xll, signing, ci]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-16T06:20:00Z }
sources:
  - id: workflow
    resource: ../.github/workflows/xll.yml
    title: The XLL workflow, Sign step
  - id: design
    resource: ../docs/design/xll.md
    title: xll.md, decisions 15 and 16
---

# Facts

* `osslsigncode` 2.8 (`apt-get install osslsigncode`, Ubuntu 24.04)
  signs the MinGW-built XLL with Authenticode on Linux: `osslsigncode sign
  -pkcs12 cert.pfx -pass ... -h sha256 -in prospicio.xll -out signed.xll`.
  Checked 2026-10-16 with a self-signed certificate (`openssl req -x509
  ... -addext extendedKeyUsage=codeSigning`): `osslsigncode verify -CAfile
  cert.pem` reports "Signature verification: ok", and the signed XLL still
  passes the fake-Excel smoke test under Wine.[^workflow]
* A PKCS#12 file in a repository secret fits a certificate from an
  organization's own CA. Publicly trusted code-signing certificates, OV as
  well as EV, must keep their keys on a hardware token or cloud HSM since
  1 June 2023 (CA/Browser Forum), so they cannot be a file in a secret; CI
  would sign through the provider's service instead.[^design]
* GitHub Actions does not show a step's own `env` to that step's `if`; the
  workflow tests a job-level `SIGN: ${{ secrets.XLL_SIGNING_PFX != '' }}`
  so that signing is skipped cleanly when no secret is set.
