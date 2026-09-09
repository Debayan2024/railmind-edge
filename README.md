# RailMind Edge

**Multi-Agent System for Predictive Signalling Safety and Driver-Assist Intelligence in Railway Networks**

## Problem Statement

- Signalling faults are detected reactively, not predicted proactively.
- No unified system exists that cross-checks track health + signal state + driver visibility + traffic congestion together.
- Existing pilots (TRI-NETRA, DOST, level-crossing cameras, Kavach) are siloed — each solves one narrow problem, none talk to each other.
- Academic DRL-based traffic optimization work exists only in isolated simulation, disconnected from the safety layer.

**Contribution:** A coordination layer — five specialized AI agents feeding a single orchestrator that cross-references signals and makes joint decisions, validated on a simulated railway corridor.

## Scope

This system does NOT control real trains, signals, or track hardware. It is a simulated, research-grade system — real ML/DL/RL models, real public + synthetic data, a working agentic orchestration layer — benchmarked against a siloed baseline. Not connected to live railway infrastructure.

## Architecture

    Orchestrator Agent (LangGraph + Gemini)
                |
Track Health | Signal Anomaly | Driver-Assist | Traffic Optimizer | Fault-Retrieval (RAG)
                |
    Simulated Railway Network (SimPy)
                |
        Streamlit Dashboard

## Status

- [x] Phase 0 — Setup
- [ ] Phase 1 — Simulation Environment
- [ ] Phase 2 — Track Health Agent
- [ ] Phase 3 — Signal Anomaly Agent
- [ ] Phase 4 — Driver-Assist Vision Agent
- [ ] Phase 5 — Traffic Optimizer Agent
- [ ] Phase 6 — Fault-Retrieval Agent (RAG)
- [ ] Phase 7 — Orchestrator
- [ ] Phase 8 — Dashboard
- [ ] Phase 9 — Evaluation
- [ ] Phase 10 — Docs/Paper

## Setup

Run these in order:

1. `uv venv --python 3.11`
2. `.venv\Scripts\activate`
3. `uv pip install -r requirements.txt`
4. `cp .env.example .env` then fill in your Gemini API key