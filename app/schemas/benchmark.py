from pydantic import BaseModel, Field

class BenchmarkInfo(BaseModel):                                                                                                                       
    name: str
    hf_id: str                                                                                                                                        
    task_type: str                                                                                                                                  
    default_metric: str                                                                                                                             
    needs_subject: bool

class BenchmarkRunCreate(BaseModel):
    benchmark: str
    subject: str | None = None # required for MMLU
    split: str = "test"                                                                                                                               
    model_id: str
    experiment_id: str | None = None # optionally add run to experiment                                                                               
    num_samples: int | None = Field(default=None, ge=1)                                                                                               
    concurrency: int = Field(default=5, ge=1, le=20)                                                                                                
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)                                                                                           
    max_tokens: int = Field(default=512, ge=1)