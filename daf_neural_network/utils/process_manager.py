"""
Process Pool Manager for Neural Network Training

This module provides a clean interface for managing ProcessPoolExecutor
with proper cleanup, CPU affinity, and job scheduling.
"""

import os
import time
import psutil
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager


class ProcessPoolManager:
    """Manages a pool of worker processes for neural network training"""

    def __init__(self, max_workers, data_queue, pause_state):
        """
        Initialize the process pool manager

        Args:
            max_workers (int): Maximum number of worker processes
            data_queue: Multiprocessing queue for communication with GUI
            pause_state: Shared state for pause/resume functionality
        """
        self.max_workers = max_workers
        self.data_queue = data_queue
        self.pause_state = pause_state
        self.executor = None
        self.future_to_job = {}
        self.jobs_queue = []
        self.submitted_jobs = set()
        self.stop_requested = False

    def start(self):
        """Start the process pool"""
        print(f"[INFO] Starting process pool with {self.max_workers} worker processes")
        print(f"[INFO] Each process will be assigned to a dedicated CPU core")

        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        return self.executor

    def add_jobs(self, jobs):
        """Add jobs to the queue"""
        self.jobs_queue.extend(jobs)
        print(f"[INFO] Added {len(jobs)} jobs to queue")

    def submit_initial_batch(self, worker_function):
        """Submit initial batch of jobs up to max_workers"""
        if not self.jobs_queue:
            print("[ERROR] No jobs to process!")
            return False

        initial_batch_size = min(self.max_workers, len(self.jobs_queue))

        for i in range(initial_batch_size):
            job_data = self.jobs_queue.pop(0)
            future = self.executor.submit(worker_function, *job_data, self.data_queue, self.pause_state)
            self.future_to_job[future] = job_data[:2]  # (job_id, model_id)
            self.submitted_jobs.add(job_data[1])  # model_id

        print(f"[INFO] Submitted {initial_batch_size} initial jobs, {len(self.jobs_queue)} jobs remaining")
        return True

    def process_completed_jobs(self, worker_function):
        """Process completed jobs and submit new ones"""
        if not self.future_to_job:
            return []

        completed_futures = [future for future in self.future_to_job.keys() if future.done()]
        results = []

        for future in completed_futures:
            job_id, model_id = self.future_to_job.pop(future)

            try:
                result = future.result()
                if result is not None:
                    results.append((model_id, result))
                    print(f"[SUCCESS] {model_id} completed successfully")
                else:
                    print(f"[ERROR] {model_id} failed")
            except Exception as e:
                print(f"[ERROR] {model_id} error: {e}")

            # Submit next job if available
            if self.jobs_queue:
                next_job_data = self.jobs_queue.pop(0)
                next_future = self.executor.submit(worker_function, *next_job_data, self.data_queue, self.pause_state)
                self.future_to_job[next_future] = next_job_data[:2]
                self.submitted_jobs.add(next_job_data[1])
                print(f"[INFO] Submitted next job {next_job_data[1]}, {len(self.jobs_queue)} jobs remaining")

        return results

    def add_dynamic_job(self, job_data, worker_function):
        """Add a job dynamically (from GUI)"""
        job_id, model_id = job_data[:2]

        if model_id in self.submitted_jobs:
            print(f"[WARNING] Skipping duplicate job {model_id}")
            return False

        # If we have available process slots, submit immediately
        if len(self.future_to_job) < self.max_workers:
            future = self.executor.submit(worker_function, *job_data, self.data_queue, self.pause_state)
            self.future_to_job[future] = (job_id, model_id)
            self.submitted_jobs.add(model_id)
            print(f"[INFO] Submitted new job {model_id} immediately")
        else:
            # Add to queue for later processing
            self.jobs_queue.append(job_data)
            self.submitted_jobs.add(model_id)
            print(f"[INFO] Added new job {model_id} to queue, {len(self.jobs_queue)} jobs waiting")

        return True

    def is_complete(self):
        """Check if all jobs are completed"""
        return not self.jobs_queue and not self.future_to_job

    def request_stop(self):
        """Request graceful stop"""
        self.stop_requested = True

    def force_shutdown(self):
        """Force shutdown all processes"""
        if not self.executor:
            return

        print("[INFO] Forcefully shutting down process pool...")
        try:
            # Cancel all pending futures
            for future in self.future_to_job:
                future.cancel()

            # Shutdown executor aggressively
            self.executor.shutdown(wait=False)

            # Force kill any remaining processes after brief wait
            time.sleep(1)

            # Get the process pool processes and terminate them
            if hasattr(self.executor, '_processes'):
                for p in self.executor._processes.values():
                    if p.is_alive():
                        print(f"[CLEANUP] Terminating process PID {p.pid}")
                        p.terminate()

                # Wait briefly then kill if still alive
                time.sleep(0.5)
                for p in self.executor._processes.values():
                    if p.is_alive():
                        print(f"[CLEANUP] Force killing process PID {p.pid}")
                        p.kill()

        except Exception as e:
            print(f"[WARNING] Error during process cleanup: {e}")

        print("[INFO] Process pool cleanup completed")


def cleanup_child_processes():
    """Final cleanup: kill any remaining python processes from this session"""
    print("[INFO] Final cleanup - checking for remaining processes...")
    try:
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)

        # Kill all child processes
        for child in parent.children(recursive=True):
            try:
                print(f"[CLEANUP] Terminating child process PID {child.pid}")
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Wait briefly then force kill if needed
        time.sleep(1)
        for child in parent.children(recursive=True):
            try:
                if child.is_running():
                    print(f"[CLEANUP] Force killing child process PID {child.pid}")
                    child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Suppress queue cleanup errors that are normal during forced shutdown
        warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")

        # Also suppress stderr output for queue errors during shutdown
        try:
            import sys
            import os
            # Redirect stderr to null to hide queue cleanup errors
            original_stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            time.sleep(0.1)  # Brief delay for any remaining output
            sys.stderr.close()
            sys.stderr = original_stderr
        except:
            pass

    except Exception as e:
        print(f"[WARNING] Error during final cleanup: {e}")


def setup_process_affinity(job_id):
    """Set up CPU affinity for worker process"""
    try:
        process = psutil.Process()
        available_cpus = list(range(psutil.cpu_count()))

        if available_cpus:
            # Assign each process to a dedicated CPU core
            assigned_cpu = available_cpus[job_id % len(available_cpus)]
            try:
                process.cpu_affinity([assigned_cpu])
                print(f"[CPU] Process assigned to CPU core {assigned_cpu}")
            except (OSError, AttributeError) as e:
                print(f"[WARNING] Could not set CPU affinity: {e}")

        # Set process priority to high for better performance
        try:
            if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
                process.nice(psutil.HIGH_PRIORITY_CLASS)
            else:
                process.nice(-5)  # Unix-like systems
        except (OSError, AttributeError):
            pass  # Continue without priority adjustment

    except Exception as e:
        print(f"[WARNING] Error setting up process: {e}")


def create_pause_check_function(pause_state, model_id):
    """Create a pause check function for a specific model"""
    def pause_check():
        if pause_state is None:
            return None  # No pause control

        try:
            # Check if this model is deleted
            if model_id in list(pause_state.get('deleted', [])):
                return True  # True = deleted

            # Check if this specific model is paused
            paused_list = list(pause_state.get('paused', []))
            if model_id in paused_list:
                return False  # False = paused

            return None  # None = continue training
        except Exception as e:
            print(f"[ERROR] Error in pause_check for {model_id}: {e}")
            return None

    return pause_check