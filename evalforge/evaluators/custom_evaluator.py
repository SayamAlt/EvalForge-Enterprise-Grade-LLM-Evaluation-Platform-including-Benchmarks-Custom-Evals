from evalforge.evaluators.base_evaluator import BaseEvaluator
from evalforge.datasets.base import EvalSample
from evalforge.providers.base import GenerationRequest, GenerationResponse

class CustomEvaluator(BaseEvaluator):
    """ Runs any dataset where num of samples have input + reference. """
    
    def __init__(self, *args, system_prompt: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt
        
    def build_prompt(self, sample: EvalSample) -> GenerationRequest:
        return GenerationRequest(
            prompt=sample.input,
            system=self.system_prompt or None,
            max_tokens=self._max_tokens,
            temperature=self._temperature
        )
        
    def extract_answer(self, response: GenerationResponse, sample: EvalSample) -> str:
        return response.text.strip()