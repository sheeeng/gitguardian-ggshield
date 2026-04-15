### Changed

- `secret scan pre-commit`, `secret scan pre-push`, `secret scan ci`, and `secret scan pre-receive` no longer block git operations when the GitGuardian server is unreachable or returns a server error. A warning is displayed and the command exits with code 0.
