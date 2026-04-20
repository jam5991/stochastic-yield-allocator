"""
train_policy.py — PPO policy training via SBX (Stable Baselines Jax).

Trains a fragmentation-aware allocation policy using PPO on the
custom Gymnasium inventory environment.
"""

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"


def train_ppo(
    env,
    total_timesteps: int = 100_000,
    seed: int = 42,
    save_path: Path | None = None,
    use_jax: bool = True,
) -> tuple:
    """Train a PPO agent on the inventory allocation environment.

    Args:
        env: Gymnasium environment instance.
        total_timesteps: Number of training timesteps.
        seed: Random seed.
        save_path: Where to save the trained model.
        use_jax: If True, use SBX (JAX backend). Falls back to SB3 if unavailable.

    Returns:
        (model, training_metrics) tuple.
    """
    if save_path is None:
        save_path = CHECKPOINT_DIR / "ppo_allocator"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training PPO for {total_timesteps:,} timesteps...")

    # Try JAX-accelerated SBX first, fall back to standard SB3
    if use_jax:
        try:
            from sbx import PPO as JaxPPO
            model = JaxPPO(
                "MlpPolicy",
                env,
                verbose=1,
                seed=seed,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
            )
            logger.info("  Using SBX (JAX-accelerated) PPO")
        except Exception as e:
            logger.warning(f"  SBX not available ({e}), falling back to SB3")
            use_jax = False

    if not use_jax:
        from stable_baselines3 import PPO as TorchPPO
        model = TorchPPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
        )
        logger.info("  Using Stable-Baselines3 (PyTorch) PPO")

    # Train with timing
    start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps)
    train_time = time.perf_counter() - start

    logger.info(f"  ✓ Training complete in {train_time:.1f}s")

    # Save model
    model.save(str(save_path))
    logger.info(f"  ✓ Model saved to {save_path}")

    training_metrics = {
        "total_timesteps": total_timesteps,
        "training_time_seconds": train_time,
        "steps_per_second": total_timesteps / train_time,
        "backend": "JAX/SBX" if use_jax else "PyTorch/SB3",
    }

    return model, training_metrics


def evaluate_rl_policy(
    model,
    env,
    n_episodes: int = 20,
    seed: int = 42,
) -> dict:
    """Evaluate a trained RL policy over multiple episodes."""
    revenues = []
    occupancies = []
    fragmentations = []
    rejection_rates = []
    latencies = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_steps = 0
        rejections = 0
        done = False

        while not done:
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)

            if int(action) == env.n_units:
                rejections += 1

            obs, _, terminated, truncated, info = env.step(int(action))
            total_steps += 1
            done = terminated or truncated

        revenues.append(info.get("total_revenue", 0.0))
        occupancies.append(info.get("occupancy_rate", 0.0))
        fragmentations.append(info.get("total_fragmentation", 0.0))
        rejection_rates.append(rejections / max(total_steps, 1))

    return {
        "policy_name": "RL (PPO + Fragmentation Penalty)",
        "mean_revenue": float(np.mean(revenues)),
        "std_revenue": float(np.std(revenues)),
        "mean_occupancy": float(np.mean(occupancies)),
        "mean_fragmentation": float(np.mean(fragmentations)),
        "mean_rejection_rate": float(np.mean(rejection_rates)),
        "revpar": float(np.mean(revenues)) / (env.n_units * env.horizon),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "mean_latency_ms": float(np.mean(latencies)),
    }


def load_model(path: Path, env, use_jax: bool = True):
    """Load a saved model checkpoint."""
    if use_jax:
        try:
            from sbx import PPO as JaxPPO
            return JaxPPO.load(str(path), env=env)
        except Exception:
            pass
    from stable_baselines3 import PPO as TorchPPO
    return TorchPPO.load(str(path), env=env)
