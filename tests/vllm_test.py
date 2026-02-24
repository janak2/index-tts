import os
import time
from random import randint, seed
# from nanovllm import LLM, SamplingParams

from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from vllm.model_executor.models.gpt2 import GPT2LMHeadModel
from vllm import ModelRegistry
from vllm.config import VllmConfig


class GPT2TestModel(GPT2LMHeadModel):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return super().generate(*args, **kwargs)


ModelRegistry.register_model("GPT2TestModel", GPT2TestModel)


def main():
    seed(0)
    num_seqs = 1
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
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]
    # uncomment the following line for vllm
    prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(
        f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s"
    )


if __name__ == "__main__":
    main()
