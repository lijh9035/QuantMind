# AlphaSAGE GPU worker

This is an opt-in service beside QuantMind. It runs the GFN miner in a separate
CUDA image and returns candidate expressions to QuantMind over an authenticated
HTTP API. The worker mounts Qlib data read-only, writes task output to its own
named volume, and joins only the private network shared with the QuantMind API.
It receives no database or Redis settings, Docker socket, or published host
port.

## Prerequisites

- NVIDIA Container Toolkit configured for Docker and a working `docker run
  --gpus all ... nvidia-smi` probe.
- AlphaSAGE source checkout at `../AlphaSAGE` relative to the QuantMind project,
  or set `ALPHASAGE_SOURCE` to its path in the local `.env`.
- QuantMind's existing Qlib provider directory, mounted read-only at `/qlib`.
  Set `QLIB_PROVIDER_PATH` if it differs from `./data/qlib/cn_data`.
- A private `ALPHASAGE_INTERNAL_TOKEN` in the local `.env`. Generate one with
  `python3 -c 'import secrets; print(secrets.token_urlsafe(32))'`; keep it out of
  source control and logs.

## Start

From the QuantMind project directory, merge the opt-in overlay and start only
the API and GPU worker (the API's existing DB and Redis dependencies are kept):

```sh
docker compose -f docker-compose.yml -f docker-compose.alphasage.yml \
  --profile alphasage up -d quantmind alphasage-gpu
```

The worker is single-slot. `/health` and all `/jobs` routes require the internal
bearer token. The durable QuantMind queue remains the source of task state; the
worker's named volume stores only execution files and result payloads. Training
uses mature labels through `mining_end - horizon` Qlib sessions. QuantMind must
evaluate candidate expressions on the independent validation window before any
factor can be persisted.

The image pins `joblib==1.3.2` and `scikit-learn==1.7.1` to match AlphaSAGE's
lock and Qlib 0.9.6's parallel data loader expectations.

The base Compose configuration does not start or build this image unless the
`alphasage` profile is selected.
