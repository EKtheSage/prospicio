### Packaging

- The release workflow builds and tests the JavaScript package on every
  run and publishes it to npm as `prospicio` on a version tag, by npm's
  trusted publishing, once the repository variable `NPM_PUBLISH` is set
  (#576, #1040; `docs/release.md`, one-time setup, step 4).
