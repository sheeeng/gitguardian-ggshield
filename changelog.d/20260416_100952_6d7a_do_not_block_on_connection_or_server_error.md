### Added

- Added a new `secret.fail_on_server_error` configuration option (default `True`), available as the `--fail-on-server-error/--no-fail-on-server-error` flag or `GITGUARDIAN_FAIL_ON_SERVER_ERROR` environment variable. When set to `False`, `secret scan pre-commit`, `secret scan pre-push`, `secret scan pre-receive`, and `secret scan ci` exit with code `0` and display a warning instead of blocking the git operation when the GitGuardian server is unreachable or returns a 5xx response. The default preserves the previous blocking behavior.

### Changed

- **Breaking**: `secret scan pre-receive` no longer fail-opens by default when the GitGuardian server returns a 5xx response. Previously the push was allowed through with a warning; now it is blocked, matching the other git hooks. Set `secret.fail_on_server_error` to `False` (or pass `--no-fail-on-server-error`) to restore the previous fail-open behavior.
