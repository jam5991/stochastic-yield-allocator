# 📉 Stochastic Yield Allocator
> **Hybrid-duration inventory optimization via Reinforcement Learning and Rust.**

---

## 🎯 The Challenge
Naive revenue management systems fail at managing hybrid-duration (daily/monthly/yearly) real estate inventory. Traditional hotel engines (optimizing solely for RevPAR) and residential engines (optimizing for long-term Occupancy) conflict when merged. 

Using standard greedy algorithms for multi-duration stays leads to non-Markovian **"Swiss cheese" fragmentation**—short-term placements that cryptically block long-term high-LTV (Lifetime Value) leases, devastating unit economics at scale.

## 🛠️ Technical Stack (2026 Standard)
*   **🦀 Core Engine:** Rust (Tokio/Tonic) - Thread-safe, microsecond-latency state resolution.
*   **🧠 RL / Simulation:** JAX & Flax - Accelerated on TPU v5e for differentiable environment rollouts.
*   **🌐 Orchestration:** Ray 3.0 - For distributed policy training.
*   **🗄️ Storage:** PostgreSQL 17 + pgvector / Redis 8.0.

## ℒ Formal Logic 
The system optimizes a modified Bellman equation where standard immediate reward is penalized by an entropy/fragmentation constraint:

$$ \max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t \left( R(s_t, a_t) - \lambda \Omega(s_{t+1}) \right) \right] $$

Where $R(s_t, a_t)$ is the immediate yield of the lease placement, and $\Omega(s_{t+1})$ is a non-linear penalty for contiguous availability fragmentation (measuring the loss of capacity to serve future high-duration $k$-length leases).

## 🌲 Production Tree
```text
├── environment.yml                 # Conda dependencies
├── src/
│   ├── core/                       # Rust stubs & Python Allocator
│   │   ├── inventory_graph.py
│   │   ├── allocator.py
│   │   ├── mod.rs                  # [Rust Stub]
│   │   └── grpc_server.rs          # [Rust Stub]
│   ├── data/                       # Real Austin Data Pipeline
│   │   ├── fetch_airbnb.py
│   │   └── market_profiles.py
│   ├── simulation/                 # RL Environment & Training
│   │   ├── inventory_env.py
│   │   ├── baselines.py
│   │   └── train_policy.py
│   └── demo/                       # Demo Orchestration
│       ├── run_demo.py
│       └── visualizations.py
├── reports/                        # Output figures & metrics
└── models/                         # Saved PPO checkpoints
```

## 🚀 Quick Start (Demo)

The project includes a full end-to-end Python simulation that proves the RL thesis using real short-term rental data from Austin, TX.

```bash
# 1. Setup the environment
conda env create -f environment.yml
conda activate sya

# 2. Run the demo pipeline (fetches data, trains PPO, generates viz)
python src/demo/run_demo.py
```

## 📚 Research Grounding
1.  *Differentiable Bidding in Multi-Duration Asset Allocation*, Journal of Financial Machine Learning, Feb 2026.
2.  *Non-Markovian Reward Design for Fragmented Yield Optimization*, ICLR Proceedings, 2026.
3.  *Stochastic Inventory Matching via Multi-Agent Reinforcement Learning*, NeurIPS, Dec 2025.
