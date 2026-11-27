### Packaging

- The npm package is staged by the release workflow (`npm stage publish`)
  and goes public when the owner approves it on npmjs.com with 2FA, so a
  CI credential alone cannot publish a version.
