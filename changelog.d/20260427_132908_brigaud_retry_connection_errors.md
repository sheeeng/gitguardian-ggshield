### Fixed

- Scans of large repositories no longer fail on a single transient network glitch. ggshield now retries connection errors (e.g. `ConnectionResetError`) and 502/503/504 responses with bounded exponential backoff.
