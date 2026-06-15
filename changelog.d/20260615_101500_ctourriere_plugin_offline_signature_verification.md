### Fixed

- `ggshield plugin install` no longer fails with "failed to refresh TUF metadata" on
  locked-down or proxied networks (most often seen on Windows). Plugin signatures are
  now always verified against the sigstore trust root bundled with ggshield's
  dependencies rather than refreshing TUF metadata over the network. Plugin identity is
  still fully enforced; only trust-root freshness now tracks the pinned sigstore version.

### Changed

- `ggshield plugin list` shows a verified plugin simply as `signed` instead of
  `signed (<signing-repository>)`. The signing identity is still recorded in the plugin
  manifest for auditing.
