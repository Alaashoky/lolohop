# HOPEFX Entry Points Reference

This document explains the purpose of each entry-point file in the project and
when to use it.

---

## Overview

| File | Purpose | When to Use |
|------|---------|-------------|
| `quickstart.py` | One-command demo / first-run helper | First time you clone the repo |
| `app.py` | FastAPI REST API server | Running the HTTP API for external access |
| `main.py` | Full integrated production system | Live / paper-trading production run |
| `main_ultimate.py` | Ultimate edition v2 with Strategy Orchestra | Advanced multi-strategy setup |
| `main_ultimate_integrated.py` | Ultimate Integrated v3 with GPU support | GPU-accelerated institutional deployment |
| `main_mcc_wrapper.py` | MCC wrapper around existing `main.py` | Enhancing an existing setup with Master Control Core |

---

## Recommended Entry Point by Use Case

### 🟢 First-time setup / demo
```bash
python quickstart.py
```
Sets up the environment, installs dependencies and runs a quick sanity-check
demo. Start here if you have just cloned the repository.

### 🌐 REST API / Web Dashboard
```bash
python app.py
# API docs available at http://localhost:5000/docs
```
Starts the FastAPI server that exposes trading operations, portfolio management,
backtesting and system health-check endpoints over HTTP.

### 🤖 Paper / Live Trading (standard)
```bash
python main.py
```
Launches the complete integrated trading system: real-time price engine, order
book, execution, brain, strategy manager and risk manager.  Defaults to
**paper-trading mode** – set `trading_enabled=true` in `.env` only when you are
ready for live execution.

### 🎻 Multi-Strategy with Strategy Orchestra
```bash
python main_ultimate.py
```
Extends the standard system with the Master Control Core (MCC) event bus and
Strategy Orchestra for coordinating multiple concurrent strategies.

### ⚡ GPU-Accelerated Institutional Deployment
```bash
python main_ultimate_integrated.py
```
Full-featured edition adding CUDA inference, Monte-Carlo risk engine, GARCH
volatility modelling, cross-exchange arbitrage and distributed multi-node
support.  Requires a CUDA-capable GPU and additional optional dependencies.

### 🔌 Enhance an Existing Setup with MCC
```python
# In your own script / existing main.py
from main_mcc_wrapper import main_with_mcc
main_with_mcc()
```
A thin wrapper you can drop into an existing codebase to add MCC capabilities
without modifying your current `main.py`.

---

## Environment Variables

All entry points read configuration from `.env`.  Copy the template and fill in
your credentials before running:

```bash
cp .env.example .env
# Edit .env – at a minimum set CONFIG_ENCRYPTION_KEY
```

See [INSTALLATION.md](./INSTALLATION.md) for a full list of required variables.
