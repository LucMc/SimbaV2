"""
Parallel experiment runner for comparing optimizer variants.

This script runs experiments in parallel to compare the performance of:
1. Standard Adam optimizer (baseline)
2. AdamS optimizer with Scheduled Weight Decay (global gradient statistics)
3. GSD optimizer with Gradient Scaled Decay (per-parameter statistics)

Usage:
    python run_paralel_swd.py --config_name online_rl --overrides env=hb_locomotion
    python run_paralel_swd.py --config_name online_rl --overrides env=dmc --seeds 0 1 2
    python run_paralel_swd.py --optimizer gsd --seeds 0 1 2  # GSD only
    python run_paralel_swd.py --optimizer adam adams gsd --seeds 0 1 2  # All optimizers

The script will launch parallel processes for each optimizer variant and seed combination.
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import hydra
import numpy as np
import omegaconf
import tqdm
from dotmap import DotMap

from scale_rl.agents import create_agent
from scale_rl.buffers import create_buffer
from scale_rl.common import WandbTrainerLogger
from scale_rl.envs import create_envs
from scale_rl.evaluation import evaluate, record_video


OptimizerType = Literal['adam', 'adams', 'gsd']


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    config_path: str
    config_name: str
    overrides: List[str]
    optimizer: OptimizerType
    seed: int
    weight_decay: float = 1e-4
    # GSD specific parameters
    decay_beta: float = 1.0
    min_grad_norm: float = 1e-8
    min_decay: float = 0.0
    max_decay: float = 1.0


def run_experiment(exp_config: ExperimentConfig) -> Dict:
    """
    Run a single experiment with the given configuration.

    Args:
        exp_config: Experiment configuration including optimizer type.

    Returns:
        Dictionary with experiment results and metadata.
    """
    args = DotMap({
        'config_path': exp_config.config_path,
        'config_name': exp_config.config_name,
        'overrides': exp_config.overrides.copy(),
    })

    # Add seed to overrides
    args.overrides.append(f'seed={exp_config.seed}')

    optimizer_name = exp_config.optimizer

    # Configure optimizer-specific settings
    if optimizer_name == 'adams':
        args.overrides.append('agent=simbaV2_swd')
        args.overrides.append(f'agent.weight_decay={exp_config.weight_decay}')
    elif optimizer_name == 'gsd':
        args.overrides.append('agent=simbaV2_gsd')
        args.overrides.append(f'agent.weight_decay={exp_config.weight_decay}')
        args.overrides.append(f'agent.decay_beta={exp_config.decay_beta}')
        args.overrides.append(f'agent.min_grad_norm={exp_config.min_grad_norm}')
        args.overrides.append(f'agent.min_decay={exp_config.min_decay}')
        args.overrides.append(f'agent.max_decay={exp_config.max_decay}')
    # else: adam - use default simbaV2 agent

    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # Initialize hydra in the subprocess
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)

    def eval_resolver(s: str):
        return eval(s, {"math": math, "__builtins__": __builtins__})

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver, replace=True)
    omegaconf.OmegaConf.resolve(cfg)

    # Set informative experiment name: {env_name}_{optimizer}_wd{weight_decay}_s{seed}
    env_name = cfg.env.env_name.replace('-', '_')
    if optimizer_name in ['adams', 'gsd']:
        weight_decay_str = f"wd{exp_config.weight_decay:.0e}".replace('-', 'm')
    else:
        weight_decay_str = "wd0"
    exp_name = f"{env_name}_{optimizer_name}_{weight_decay_str}_s{exp_config.seed}"
    cfg.exp_name = exp_name

    # Set group_name to optimizer name for easy filtering in wandb
    cfg.group_name = optimizer_name

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Create environments
    train_env, eval_env = create_envs(**cfg.env)
    observation_space = train_env.observation_space
    action_space = train_env.action_space

    # Create buffer
    buffer = create_buffer(
        observation_space=observation_space, action_space=action_space, **cfg.buffer
    )
    buffer.reset()

    # Create agent
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        cfg=cfg.agent,
    )

    # Training loop
    logger = WandbTrainerLogger(cfg)

    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    save_path = script_dir + "/" + cfg.save_path
    if cfg.load_path:
        load_path = script_dir + "/" + cfg.load_path
        agent.load(load_path)

    # Initial evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
    logger.update_metric(**eval_info)
    logger.log_metric(step=0)
    logger.reset()

    # Training
    update_step = 0
    update_counter = 0
    observations, env_infos = train_env.reset()
    timestep = None

    results = {
        'optimizer': optimizer_name,
        'seed': exp_config.seed,
        'eval_rewards': [],
        'final_reward': 0.0,
    }

    for interaction_step in tqdm.tqdm(
        range(1, int(cfg.num_interaction_steps + 1)),
        smoothing=0.1,
        desc=f"{optimizer_name} seed={exp_config.seed}"
    ):
        # Collect data
        if timestep:
            actions = agent.sample_actions(
                interaction_step, prev_timestep=timestep, training=True
            )
        if buffer.can_sample() is False:
            actions = train_env.action_space.sample()

        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(
            actions
        )
        next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        timestep = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }
        buffer.add(timestep)
        timestep["next_observation"] = next_observations
        observations = next_observations

        if buffer.can_sample():
            # Update network
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                batch = buffer.sample()
                update_info = agent.update(update_step, batch)
                logger.update_metric(**update_info)
                update_counter -= 1
                update_step += 1

            # Evaluation
            if interaction_step % cfg.evaluation_per_interaction_step == 0:
                eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes)
                logger.update_metric(**eval_info)
                results['eval_rewards'].append(eval_info.get('eval/reward_mean', 0.0))

            # Metrics
            if interaction_step % cfg.metrics_per_interaction_step == 0:
                batch = buffer.sample()
                metrics_info = agent.get_metrics(batch, update_info)
                if metrics_info:
                    logger.update_metric(**metrics_info)

            # Video recording
            if interaction_step % cfg.recording_per_interaction_step == 0:
                video_info = record_video(agent, eval_env, cfg.num_record_episodes)
                logger.update_metric(**video_info)

            # Logging
            if interaction_step % cfg.logging_per_interaction_step == 0:
                env_step = interaction_step * cfg.action_repeat * cfg.num_train_envs
                logger.log_metric(step=env_step)
                logger.reset()

            # Checkpointing
            if interaction_step % cfg.save_checkpoint_per_interaction_step == 0:
                agent.save(save_path)

            # Save buffer
            if interaction_step % cfg.save_buffer_per_interaction_step == 0:
                buffer.save(save_path)

    train_env.close()
    eval_env.close()

    if results['eval_rewards']:
        results['final_reward'] = results['eval_rewards'][-1]

    return results


def run_parallel_experiments(
    config_path: str,
    config_name: str,
    overrides: List[str],
    seeds: List[int],
    optimizers: List[OptimizerType],
    weight_decay: float = 1e-4,
    decay_beta: float = 1.0,
    min_grad_norm: float = 1e-8,
    min_decay: float = 0.0,
    max_decay: float = 1.0,
    max_workers: Optional[int] = None,
) -> List[Dict]:
    """
    Run experiments in parallel for specified optimizers across multiple seeds.

    Args:
        config_path: Path to config directory.
        config_name: Name of the config file.
        overrides: List of hydra overrides.
        seeds: List of random seeds to use.
        optimizers: List of optimizer types to run ('adam', 'adams', 'gsd').
        weight_decay: Weight decay for AdamS/GSD experiments.
        decay_beta: GSD: EMA coefficient for gradient magnitude smoothing.
        min_grad_norm: GSD: Minimum gradient norm.
        min_decay: GSD: Minimum effective decay.
        max_decay: GSD: Maximum effective decay.
        max_workers: Maximum number of parallel workers (default: CPU count).

    Returns:
        List of result dictionaries from all experiments.
    """
    experiments = []

    for seed in seeds:
        for optimizer in optimizers:
            experiments.append(ExperimentConfig(
                config_path=config_path,
                config_name=config_name,
                overrides=overrides.copy(),
                optimizer=optimizer,
                seed=seed,
                weight_decay=weight_decay,
                decay_beta=decay_beta,
                min_grad_norm=min_grad_norm,
                min_decay=min_decay,
                max_decay=max_decay,
            ))

    results = []

    if max_workers == 1:
        # Sequential execution for debugging
        for exp in experiments:
            result = run_experiment(exp)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_experiment, exp): exp for exp in experiments}
            for future in as_completed(futures):
                exp = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Completed: {result['optimizer']} seed={result['seed']}, "
                          f"final_reward={result['final_reward']:.2f}")
                except Exception as e:
                    print(f"Error in {exp.optimizer}, seed={exp.seed}: {e}")

    return results


def print_comparison_summary(results: List[Dict]) -> None:
    """Print a summary comparing optimizer results."""
    optimizer_groups = {}
    for r in results:
        opt = r['optimizer']
        if opt not in optimizer_groups:
            optimizer_groups[opt] = []
        optimizer_groups[opt].append(r)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 60)

    optimizer_means = {}

    for opt_name, opt_results in sorted(optimizer_groups.items()):
        rewards = [r['final_reward'] for r in opt_results]
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        optimizer_means[opt_name] = mean_reward

        opt_display = {
            'adam': 'Adam (baseline)',
            'adams': 'AdamS (Scheduled Weight Decay - global)',
            'gsd': 'GSD (Gradient Scaled Decay - per-param)',
        }.get(opt_name, opt_name)

        print(f"\n{opt_display}:")
        print(f"  Seeds: {[r['seed'] for r in opt_results]}")
        print(f"  Final rewards: {rewards}")
        print(f"  Mean: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Print improvements relative to baseline
    if 'adam' in optimizer_means:
        baseline = optimizer_means['adam']
        print(f"\nImprovements vs Adam baseline:")
        for opt_name, mean in optimizer_means.items():
            if opt_name != 'adam':
                improvement = mean - baseline
                print(f"  {opt_name}: {improvement:+.2f}")

    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(
        description="Run parallel experiments comparing optimizer variants",
        allow_abbrev=False
    )
    parser.add_argument(
        "--config_path", type=str, default="./configs",
        help="Path to config directory"
    )
    parser.add_argument(
        "--config_name", type=str, default="online_rl",
        help="Name of the config file"
    )
    parser.add_argument(
        "--overrides", action="append", default=[],
        help="Hydra config overrides"
    )
    parser.add_argument(
        "--seeds", type=int, nargs='+', default=[0, 1, 2],
        help="Random seeds for experiments"
    )
    parser.add_argument(
        "--optimizer", type=str, nargs='+', default=['adam', 'adams'],
        choices=['adam', 'adams', 'gsd'],
        help="Optimizer(s) to run (default: adam adams)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4,
        help="Weight decay coefficient for AdamS/GSD"
    )
    # GSD specific arguments
    parser.add_argument(
        "--decay_beta", type=float, default=1.0,
        help="GSD: EMA coefficient for gradient magnitude smoothing (1.0 = no additional EMA)"
    )
    parser.add_argument(
        "--min_grad_norm", type=float, default=1e-8,
        help="GSD: Minimum gradient norm to avoid division by zero"
    )
    parser.add_argument(
        "--min_decay", type=float, default=0.0,
        help="GSD: Minimum effective decay after scaling"
    )
    parser.add_argument(
        "--max_decay", type=float, default=1.0,
        help="GSD: Maximum effective decay after scaling"
    )
    parser.add_argument(
        "--max_workers", type=int, default=None,
        help="Maximum parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run experiments sequentially (for debugging)"
    )

    args = parser.parse_args()

    max_workers = 1 if args.sequential else args.max_workers

    print(f"Starting parallel optimizer comparison experiments")
    print(f"  Optimizers: {args.optimizer}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Weight decay: {args.weight_decay}")
    if 'gsd' in args.optimizer:
        print(f"  GSD decay_beta: {args.decay_beta}")
        print(f"  GSD min_grad_norm: {args.min_grad_norm}")
        print(f"  GSD min/max_decay: [{args.min_decay}, {args.max_decay}]")
    print(f"  Max workers: {max_workers or 'auto'}")

    results = run_parallel_experiments(
        config_path=args.config_path,
        config_name=args.config_name,
        overrides=args.overrides,
        seeds=args.seeds,
        optimizers=args.optimizer,
        weight_decay=args.weight_decay,
        decay_beta=args.decay_beta,
        min_grad_norm=args.min_grad_norm,
        min_decay=args.min_decay,
        max_decay=args.max_decay,
        max_workers=max_workers,
    )

    print_comparison_summary(results)
