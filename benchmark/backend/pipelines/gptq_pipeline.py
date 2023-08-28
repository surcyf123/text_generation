from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, logging, pipeline


class GPTQInference:
    def __init__(self, model_dir: str, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.pipeline, self.model = self.load_model(model_dir, model_name)

    def generate(self, prompt):
        # use prompt template to normalize LLMs answers
        prompt_template = f"""Human{prompt}
        ### Assistant:"""

        return self.pipeline(prompt_template)[0]["generated_text"]

    def load_model(self, model_dir: str, model_name: str):
        quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=False)

        model = AutoGPTQForCausalLM.from_quantized(
            model_dir,
            use_safetensors=True,
            model_basename=model_name,
            device="cuda:0",
            use_triton=False,
            strict=False,
            quantize_config=quantize_config,
            trust_remote_code=True,
        )

        # Prevent printing spurious transformers error when using pipeline with AutoGPTQ
        logging.set_verbosity(logging.CRITICAL)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            # temperature=0.7,
            # top_p=0.95,
            repetition_penalty=1.15,
        )

        return pipe, model

    def unload_model(self):
        del self.tokenizer
        del self.pipeline
        del self.model
