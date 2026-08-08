"""
Celery task definitions — implemented in Sprint 4.

Tasks planned:
  run_evaluation_task(run_id, dataset_id, model_id, config)
  run_benchmark_task(run_id, benchmark_name, model_id, n_samples)
  run_comparison_task(run_id, model_ids, dataset_id)
"""
from app.workers.celery_app import celery_app

@celery_app.task(name="run_evaluation", bind=True)
def run_evaluation_task(self, run_id: str, dataset_id: str, model_id: str) -> dict:
    """ [Sprint 4] Run a custom dataset evaluation in the background. """
    raise NotImplementedError("Evaluation Engine implemented in Sprint 4.")

@celery_app.task(name="run_benchmark", bind=True)
def run_benchmark_task(self, run_id: str, benchmark: str, model_id: str) -> dict:
    """ [Sprint 4] Run a standard benchmark (MMLU, GSM8K, etc.) in the background. """
    raise NotImplementedError("Benchmark Evaluator implemented in Sprint 4.")