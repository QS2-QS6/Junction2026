# Security Review

## Scope and validation

Read-only review of the visualizer HTTP server, recording endpoints, local
credential handling, Tiger Data importer, and gesture training code. The
automated Python test suites passed (14 tests total), and the Python source
compiled successfully. No live Rokoko device, Tiger Data connection, or Verda
instance was used during this review.

## Findings

| # | Severity | File | Lines | Vulnerability | Confidence |
|---|----------|------|-------|---------------|------------|
| 1 | 🟡 MEDIUM | `glove_viz/server.py` | 229-289, 326-346 | The live `/stream` endpoint and recording start/stop endpoints have no authentication or Origin/CSRF checks. If the server is bound to a non-loopback address with `--http-host`, any reachable client can view motion data and start or stop recordings. The default bind is loopback, so the risk applies when remote binding is enabled. | 8/10 |

## Recommendation

Keep the HTTP server bound to `127.0.0.1` unless remote access is specifically
needed. If remote access is required, add authentication and request-origin /
CSRF protections for recording controls, and restrict network access with a
firewall or private tunnel.
