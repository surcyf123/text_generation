import gc
import os

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, logging, pipeline


class GPTQInference:
    def __init__(self, model_dir: str, model_name: str, group_size: int, gpu_id:str="cuda:0"):
        self.model_name = os.path.basename(model_dir)
        if self.model_name == "h2ogpt-oasst1-512-30B-GPTQ":
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, unk_token="<unk>", bos_token="<s>", eos_token="<\s>")
        else: 
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.pipeline, self.model = self.load_model(model_dir, model_name, group_size, gpu_id)

    def generate(self, prompt):
        # use prompt template to normalize LLMs answers
        prompt_template = f"""
        Demonstrate a potential experiment while utilizing and enumerating the scientific method clearly and explain every step for a potential theory of the following context.
        ### USER: {prompt}
        <\s> 

        ASSISTANT:"""

        if self.model_name == "Llama-2-13B-GPTQ":
            prompt_template = f"""{prompt}"""

        if self.model_name == "h2ogpt-oasst1-512-30B-GPTQ":
            prompt_template = f"""<human>: {prompt}
            <bot>:"""
        if self.model_name == "WizardLM-30B-Uncensored-GPTQ":
            prompt_template = f"""{prompt}
            ### Response:"""
        if self.model_name == "guanaco-33B-GPTQ":
            prompt_template = f"""### Human: {prompt}
            ### Assistant:"""
        if self.model_name == "Nous-Hermes-13B-GPTQ":
            prompt_template = f"""### Instruction:
            ### Response:"""
        if self.model_name == "Metharme-13b-4bit-GPTQ":
            prompt_template = f"""
            <|system|>Demonstrate a potential experiment while utilizing and enumerating the scientific method clearly and explain every step for a potential theory of the following context.
            <|user|>{prompt}
            <|model|>"""

        return self.pipeline(prompt_template)[0]["generated_text"]

    def generate_batch(self, prompts, batch_size=8):
        prompt_templates = [f"""
            Demonstrate a potential experiment while utilizing and enumerating the scientific method clearly and explain every step for a potential theory of the following context.
            ### USER: {prompt}
            <\s> 

            ASSISTANT:"""
            for prompt in prompts
        ]

        if self.model_name == "Llama-2-13B-GPTQ":
            prompt_templates = [f"""{prompt}""" for prompt in prompts]

        if self.model_name == "h2ogpt-oasst1-512-30B-GPTQ":
            prompt_templates = [f"""<human>: {prompt}
            <bot>:""" 
            for prompt in prompts]

        if self.model_name == "WizardLM-30B-Uncensored-GPTQ":
            prompt_templates = [f"""{prompt}
            ### Response:"""
            for prompt in prompts]

        if self.model_name == "guanaco-33B-GPTQ":
            prompt_templates = [f"""### Human: {prompt}
            ### Assistant:"""
            for prompt in prompts]

        if self.model_name == "Nous-Hermes-13B-GPTQ":
            prompt_templates = [f"""### Instruction:
            ### Response:"""
            for prompt in prompts]

        if self.model_name == "Metharme-13b-4bit-GPTQ":
            prompt_templates = [f"""
            <|system|>Demonstrate a potential experiment while utilizing and enumerating the scientific method clearly and explain every step for a potential theory of the following context.
            <|user|>{prompt}
            <|model|>"""
            for prompt in prompts]
            
        results = self.pipeline(prompt_templates, batch_size=batch_size)
        return [result[0]["generated_text"] for result in results]

    def load_model(self, model_dir: str, model_name: str, group_size: int=128, gpu_id: str="cuda:0"):
        quantize_config = BaseQuantizeConfig(bits=4, group_size=group_size, desc_act=False)

        model = AutoGPTQForCausalLM.from_quantized(
            model_dir,
            use_safetensors=True,
            model_basename=model_name,
            device=gpu_id,
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
            max_new_tokens=1024,
            min_new_tokens=256, #transformers.GenerationConfig
            #temperature=0.7,
            # top_p=0.95,
            repetition_penalty=1.15,
            #do_sample=True
            
        )

        pipe.tokenizer.pad_token_id = model.config.eos_token_id

        return pipe, model

    def unload_model(self):
        del self.tokenizer
        del self.pipeline
        del self.model
        gc.collect()
