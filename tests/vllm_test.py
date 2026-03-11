import time
from random import randint, seed
from typing import Optional

import torch
from huggingface_hub import snapshot_download

# from nanovllm import LLM, SamplingParams
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.config import VllmConfig
from vllm.model_executor.models.gpt2 import GPT2LMHeadModel
from vllm.sequence import IntermediateTensors


class GPT2TestModel(GPT2LMHeadModel):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        print(inputs_embeds.shape if inputs_embeds is not None else "None")
        print(input_ids.shape if input_ids is not None else "None")
        return super().forward(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )


ModelRegistry.register_model("GPT2TestModel", GPT2TestModel)


def main():
    seed(0)
    num_seqs = 5
    max_input_len = 1024
    max_ouput_len = 1024

    path = snapshot_download(
        "openai-community/gpt2",
        ignore_patterns=["*.tflite", "*.onnx", "*.msgpack", "*.ot", "*.h5"],
    )
    llm = LLM(
        path,
        max_model_len=1024,
        max_num_seqs=num_seqs,
        hf_overrides={"architectures": ["GPT2TestModel"]},
        skip_tokenizer_init=False,
        enable_prompt_embeds=True,
        dtype="float16",
        gpu_memory_utilization=0.8,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(100)],
        [randint(0, 10000) for _ in range(150)],
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(len(prompt_token_ids))
    ]
    # uncomment the following line for vllm
    prompt_token_ids = [
        dict(prompt_token_ids=p, prompt_embeds=torch.randn(1, len(p), 768))
        for p in prompt_token_ids
    ]

    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    print(f"Time: {t:.2f}s")
    print(len(outputs[0].outputs[0].token_ids))


if __name__ == "__main__":
    main()
