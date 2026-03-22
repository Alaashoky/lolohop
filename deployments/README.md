# `deployments/` – Platform-Specific Deployment Artefacts

This folder contains **platform-specific deployment configurations**, organised
into sub-folders so that each environment can be managed independently.

| Sub-folder | Purpose |
|------------|---------|
| `docker/` | Docker build files for production and development images |
| `k8s/` | Kubernetes manifests (Namespace, Deployment, Service, HPA, ConfigMap) |

## When to use `deployments/`

Use this folder when you are deploying to a **specific platform** and need
granular control over each configuration file, or when you are managing
multiple environments (staging, production) within the same repo.

## Difference from `deployment/`

| | `deployments/` | `deployment/` |
|-|----------------|---------------|
| Contents | Separated sub-folders per platform | Combined single-file configs |
| Use when | Platform-specific / multi-environment | Simple one-command deploy |

### `docker/`
- `Dockerfile` – Production image
- `Dockerfile.dev` – Development image with hot-reload
- `entrypoint.sh` – Container startup script

### `k8s/`
- `namespace.yaml` – Kubernetes namespace definition
- `deployment.yaml` – Pod spec and replica settings
- `service.yaml` – ClusterIP / LoadBalancer service
- `hpa.yaml` – Horizontal Pod Autoscaler
- `configmap.yaml` – Non-sensitive environment configuration

See also [`deployment/README.md`](../deployment/README.md).
