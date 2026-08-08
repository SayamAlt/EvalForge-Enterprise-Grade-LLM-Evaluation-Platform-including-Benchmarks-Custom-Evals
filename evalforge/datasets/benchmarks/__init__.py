import asyncio
from evalforge.datasets.base import EvalSample                                                               
from evalforge.datasets.loaders.huggingface_loader import load_from_huggingface                                                                     
                                                                                                                                                    
BENCHMARK_CATALOG = {
    "mmlu": {"hf_id": "cais/mmlu",            "task_type": "multiple_choice", "default_metric": "choice_match",  "needs_config": True},         
    "gsm8k": {"hf_id": "openai/gsm8k",          "task_type": "generation",      "default_metric": "answer_match",  "hf_config": "main"},         
    "hellaswag": {"hf_id": "Rowan/hellaswag",       "task_type": "multiple_choice", "default_metric": "choice_match"},                               
    "humaneval": {"hf_id": "openai/openai_humaneval","task_type": "code",            "default_metric": "code_match",   "hf_split": "test"},
    "hle":       {"hf_id": "cais/hle",              "task_type": "generation",      "default_metric": "answer_match",  "hf_split": "test"},
}                                                                                                                                                     
                                                                                                                                                    
async def load_benchmark(name: str, subject: str | None = None, split: str = "test", n_samples: int | None = None) -> list[EvalSample]:                                                   
    if name not in BENCHMARK_CATALOG:                                                                                                                 
        raise ValueError(f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_CATALOG)}")
                                                         
    cfg = BENCHMARK_CATALOG[name]                                                                                                                     
    hf_config = subject if cfg.get("needs_config") else cfg.get("hf_config")
    dataset_name = f"{name}_{subject}" if subject else name    
                                                                                           
    return await asyncio.to_thread(                                                                                                                 
        load_from_huggingface,                                                                                                                        
        cfg["hf_id"], dataset_name,                                                                                                                 
        hf_config=hf_config, hf_split=split, n_samples=n_samples,                                                                                     
    )