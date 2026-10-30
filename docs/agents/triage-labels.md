# Triage Labels

The triage skill uses five canonical state roles. This file maps them to the state strings used in local issue files.

| Canonical role | Local state | Meaning |
|---|---|---|
| `needs-triage` | `needs-triage` | The maintainer needs to evaluate the issue. |
| `needs-info` | `needs-info` | Waiting for more information from the reporter. |
| `ready-for-agent` | `ready-for-agent` | Fully specified and ready for an agent. |
| `ready-for-human` | `ready-for-human` | Requires human implementation. |
| `wontfix` | `wontfix` | Will not be actioned. |

When a skill mentions one of these state roles, use the corresponding local state string. Each triaged issue should also have exactly one category role: `bug` or `enhancement`.
