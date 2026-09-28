# Optional downstream notification

The **Notify private downstream** GitHub Actions workflow sends a pipeline
trigger request after main changes. It checks out no code and reads no
downstream repository contents.

To configure a GitLab downstream, add Actions secrets:

- `GITLAB_SYNC_TRIGGER_URL`: the complete GitLab project pipeline trigger URL.
- `GITLAB_SYNC_TRIGGER_TOKEN`: a GitLab pipeline trigger token.

The downstream must accept a pipeline trigger for `main` and provide its own
integration and review workflow. Run this workflow manually first; then set
Actions variable `PRIVATE_SYNC_ENABLED` to `true` to enable pushes to main.
Leaving the variable unset disables automatic notifications. Missing secrets
cause a manual run to fail. Never commit credentials.

A successful notification confirms acceptance of the request only. Integration
results and merge approval belong to the downstream. The trigger endpoint must
be reachable from GitHub-hosted runners. Revoke the trigger token to disable
both automatic and manual requests.
