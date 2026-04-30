### Fixed

- Skip OS keyring access at startup when `GITGUARDIAN_API_KEY` is set in the environment (or in a `.env` file). This avoids redundant keychain unlock prompts on systems using multiple ggshield intances.
