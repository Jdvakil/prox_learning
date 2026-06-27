from datetime import datetime
import logging
import os
import pprint
import time
from pathlib import Path

from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner, setup_house_dirs, cleanup_context, mp_context
from molmo_spaces.utils.mp_logging import get_worker_logger, worker_stdout_context
from molmo_spaces.utils.profiler_utils import DatagenProfiler

def custom_house_processing_worker(
    worker_id: int,
    exp_config,
    house_indices: list[int],
    samples_per_house: int,
    shutdown_event,
    counter_lock,
    house_counter,
    success_count,
    total_count,
    completed_houses,
    skipped_houses,
    house_results,
    max_allowed_sequential_task_sampler_failures: int = 10,
    max_allowed_sequential_rollout_failures: int = 10,
    max_allowed_sequential_irrecoverable_failures: int = 5,
    preloaded_policy = None,
    filter_for_successful_trajectories: bool = False,
    runner_class = None,
):
    worker_logger = get_worker_logger(worker_id)
    if hasattr(exp_config, "datagen_profiler") and exp_config.datagen_profiler:
        datagen_profiler = DatagenProfiler(logger=worker_logger, enabled=True)
    else:
        datagen_profiler = None

    num_sequential_irrecoverable_failures = 0
    task_sampler = exp_config.task_sampler_config.task_sampler_class(exp_config)
    task_sampler.set_datagen_profiler(datagen_profiler)

    with worker_stdout_context(worker_logger, worker_id):
        try:
            while True:
                if shutdown_event.is_set():
                    worker_logger.info(f"Worker {worker_id} received shutdown signal, cleaning up...")
                    break

                with counter_lock:
                    target_successes = getattr(runner_class, "target_successes", 83)
                    if success_count.value >= target_successes:
                        worker_logger.info(
                            f"Worker {worker_id} stopping early because target successful trajectories "
                            f"({target_successes}) reached (current successes: {success_count.value})"
                        )
                        break

                    if house_counter.value >= len(house_indices):
                        break  # No more houses to process
                    house_idx = house_counter.value
                    current_house_id = house_indices[house_idx]
                    house_counter.value += 1

                batch_file = Path(exp_config.output_dir) / f"house_{current_house_id}" / "trajectories_batch_1_of_1.h5"
                if batch_file.exists():
                    worker_logger.info(f"Worker {worker_id} house {current_house_id} already has completed trajectory. Skipping.")
                    house_results.append({
                        "house_id": current_house_id,
                        "index": house_idx,
                        "status": "Success (Existing)",
                        "successes": 1,
                        "attempts": 1
                    })
                    with counter_lock:
                        total_count.value += 1
                        completed_houses.value += 1
                    continue

                worker_logger.info(
                    f"Worker {worker_id} starting house {current_house_id} (index {house_idx}/{len(house_indices)})"
                )

                house_success_count, house_total_count, irrecoverable = (
                    runner_class.process_single_house(
                        worker_id,
                        worker_logger,
                        current_house_id,
                        exp_config,
                        samples_per_house,
                        shutdown_event,
                        task_sampler,
                        preloaded_policy,
                        max_allowed_sequential_task_sampler_failures,
                        max_allowed_sequential_rollout_failures,
                        filter_for_successful_trajectories=filter_for_successful_trajectories,
                        runner_class=runner_class,
                        datagen_profiler=datagen_profiler,
                    )
                )

                # Determine status and record results
                if house_total_count == 0:
                    status = "Skipped"
                elif house_success_count > 0:
                    status = "Success"
                else:
                    status = "Failed"

                house_results.append({
                    "house_id": current_house_id,
                    "index": house_idx,
                    "status": status,
                    "successes": house_success_count,
                    "attempts": house_total_count
                })

                with counter_lock:
                    success_count.value += house_success_count
                    total_count.value += house_total_count
                    if house_total_count > 0:
                        completed_houses.value += 1
                    else:
                        skipped_houses.value += 1

                if irrecoverable:
                    num_sequential_irrecoverable_failures += 1
                    if num_sequential_irrecoverable_failures >= max_allowed_sequential_irrecoverable_failures:
                        worker_logger.error(
                            f"Worker {worker_id} encountered {num_sequential_irrecoverable_failures} "
                            "sequential irrecoverable failures. Exiting worker."
                        )
                        break
                else:
                    num_sequential_irrecoverable_failures = 0

            worker_logger.info(f"Worker {worker_id} completed processing assigned houses")
        finally:
            if datagen_profiler is not None:
                datagen_profiler.log_worker_summary()
            if task_sampler is not None:
                task_sampler.close()

class CustomParallelRolloutRunner(ParallelRolloutRunner):
    target_successes = 83

    def __init__(self, exp_config):
        super().__init__(exp_config)
        self.manager = mp_context.Manager()
        self.house_results = self.manager.list()

    def run(self, preloaded_policy=None):
        # Count existing successes on disk
        existing_h5 = list(Path(self.config.output_dir).glob("**/trajectories_batch_*.h5"))
        self.success_count.value = len(existing_h5)

        total_expected_episodes = self.total_houses * self.samples_per_house
        self.logger.info(
            f"Starting Custom parallel rollout of {self.total_houses} houses "
            f"with {self.samples_per_house} episodes each ({total_expected_episodes} total episodes) "
            f"using {self.config.num_workers} workers"
        )
        self.logger.info(f"Resuming with {self.success_count.value} existing successful trajectories already on disk.")

        self.logger.info("Evaluation configuration:")
        self.logger.info(pprint.pformat(self.config.model_dump()))
        self.config.save_config(output_dir=Path(self.config.output_dir))

        start_time = time.time()

        # Call custom multiprocessing workers
        processes = []
        for worker_id in range(self.config.num_workers):
            p = mp_context.Process(
                target=custom_house_processing_worker,
                args=(
                    worker_id,
                    self.config,
                    self.house_indices,
                    self.samples_per_house,
                    self.shutdown_event,
                    self.counter_lock,
                    self.house_counter,
                    self.success_count,
                    self.total_count,
                    self.completed_houses,
                    self.skipped_houses,
                    self.house_results,
                    self.max_allowed_sequential_task_sampler_failures,
                    self.max_allowed_sequential_rollout_failures,
                    self.max_allowed_sequential_irrecoverable_failures,
                    preloaded_policy,
                    self.config.filter_for_successful_trajectories,
                    type(self),
                ),
            )
            p.start()
            processes.append(p)

        # Monitor processes
        try:
            while any(p.is_alive() for p in processes):
                time.sleep(1.0)
                # Check if target reached to set shutdown event
                if self.success_count.value >= self.target_successes:
                    if not self.shutdown_event.is_set():
                        self.logger.info(f"Target of {self.target_successes} successes reached. Signaling shutdown...")
                        self.shutdown_event.set()
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user. Shutting down...")
            self.shutdown_event.set()

        # Join processes
        for p in processes:
            p.join()

        success_count_val = self.success_count.value
        total_count_val = self.total_count.value
        completed_houses_val = self.completed_houses.value
        skipped_houses_val = self.skipped_houses.value

        success_rate = success_count_val / total_count_val if total_count_val > 0 else 0.0
        elapsed_time = time.time() - start_time

        self.logger.info(
            f"Completed {completed_houses_val} houses, skipped {skipped_houses_val} houses"
        )
        self.logger.info(f"Success count: {success_count_val}, Total count: {total_count_val}")
        self.logger.info(f"Success rate: {success_rate * 100:.2f}%")
        self.logger.info(f"Total time elapsed: {elapsed_time:.2f}s")

        # Print per-house status summary table
        print("\n=== PER-HOUSE LOGGING SUMMARY ===", flush=True)
        print("| Index | House ID | Status | Successes | Attempts |", flush=True)
        print("|-------|----------|--------|-----------|----------|", flush=True)
        for res in sorted(list(self.house_results), key=lambda x: x['index']):
            print(f"| {res['index']} | {res['house_id']} | {res['status']} | {res['successes']} | {res['attempts']} |", flush=True)
        print("==================================\n", flush=True)

        return success_count_val, total_count_val

def main():
    base_dir = Path('/mnt/d/Machines virtueles/prox_learning/assets/datagen/pick_and_place_skin_open_front_cluttered_reach_pilot_v2')
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / run_id
    print(f"Creating new run dir: {output_dir}", flush=True)

    cfg = FrankaSkinOpenFrontClutteredReachPickAndPlacePilotConfig()
    cfg.num_workers = 8
    cfg.seed = 42
    cfg.use_wandb = False
    cfg.collision_free_pose_limit = 30
    cfg.task_sampler_config.house_inds = list(range(300))
    cfg.task_sampler_config.samples_per_house = 1
    cfg.task_sampler_config.max_allowed_sequential_task_sampler_failures = 2000
    cfg.task_sampler_config.max_allowed_sequential_rollout_failures = 2000
    cfg.task_sampler_config.max_total_attempts_multiplier = 25
    cfg.task_sampler_config.max_asset_failures = 10_000
    cfg.task_sampler_config.max_robot_placement_attempts = 80
    cfg.policy_config.filter_colliding_grasps = False
    cfg.filter_for_successful_trajectories = True
    cfg.output_dir = output_dir
    cfg.save_config()
    print('output_dir=', cfg.output_dir, flush=True)

    CustomParallelRolloutRunner.target_successes = 83

    success_count, total_count = CustomParallelRolloutRunner(cfg).run()
    print('Final Success count:', success_count, 'Total count:', total_count, flush=True)

if __name__ == '__main__':
    main()