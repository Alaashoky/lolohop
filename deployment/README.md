# `deployment/` – Ready-to-Run Deployment Configs

This folder contains **ready-to-run, self-contained deployment artefacts** –
everything you need to spin up the trading bot quickly.

| File | Purpose |
|------|---------|
| `Dockerfile` | Single-container Docker image |
| `docker-compose.yml` | Multi-service stack (app + Postgres + Redis) |
| `k8s_config.yaml` | Kubernetes manifest (Deployment + Service) |

## When to use `deployment/`

Use this folder for **quick deployments** when you want a single `docker-compose up`
or `kubectl apply` command to get the system running.

## Difference from `deployments/`

| | `deployment/` | `deployments/` |
|-|---------------|----------------|
| Contents | Combined, single-file configs | Separated sub-folders per platform |
| Use when | Simple one-command deploy | Platform-specific or team environments |

See also [`deployments/README.md`](../deployments/README.md).
