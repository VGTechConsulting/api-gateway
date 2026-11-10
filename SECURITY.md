# Security policy

## Supported versions

Only the latest release is supported.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability
reporting ("Security" tab → "Report a vulnerability"). Do not open a public issue.

## Deployment reminder

The gate holds provider API keys. It binds to `127.0.0.1` by default and the
compose file publishes it on loopback only. Do not expose it to a public network:
it has no SSO, no TLS termination and no admin authentication beyond host access.
Environment variables are visible in `docker inspect`; production setups should
use a secrets manager.

The read-only report UI (`api-gateway ui`, compose service `api-gateway-ui`) has
**no authentication**. Anyone who can reach it sees owners, teams, apps, models,
key prefixes and spend. It binds to `127.0.0.1` by default and compose publishes
it on loopback only; never publish it on a network interface. It holds no
provider keys and opens the database read-only.
