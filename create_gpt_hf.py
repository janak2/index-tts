from omegaconf import OmegaConf
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.checkpoint import load_checkpoint
import os
import torch
import os
import time
from random import randint, seed
# from nanovllm import LLM, SamplingParams

from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from vllm.model_executor.models.gpt2 import GPT2LMHeadModel
from vllm import ModelRegistry
from vllm.config import VllmConfig
import torch
from typing import Optional
from vllm.sequence import IntermediateTensors
from indextts.gpt.model_v2_vllm import GPT2LMHeadModel as GPT2InferenceModel
import argparse


def push_gpt_hf(
    cfg_path: str,
    model_dir: str,
    repo_id: str = "janak22/index-tts-gpt",
    use_deepspeed: bool = False,
    use_fp16: bool = False,
    use_accel: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(cfg_path)
    gpt = UnifiedVoice(**cfg.gpt, use_accel=use_accel)
    gpt_path = os.path.join(model_dir, cfg.gpt_checkpoint)
    load_checkpoint(gpt, gpt_path)
    gpt = gpt.to(device)

    gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=use_fp16)

    # Save inference model as Hugging Face checkpoint to huggingface hub
    gpt.inference_model.config.max_mel_seq_len = 1818
    gpt.inference_model.push_to_hub(repo_id, safe_serialization=False)


ModelRegistry.register_model("GPT2InferenceModel", GPT2InferenceModel)


def load_gpt_hf(
    repo_id: str,
):
    path = snapshot_download(
        repo_id,
        ignore_patterns=["*.tflite", "*.onnx", "*.msgpack", "*.ot", "*.h5"],
    )
    llm = LLM(
        path,
        max_model_len=1024,
        max_num_seqs=1,
        # hf_overrides={"architectures": ["GPT2TestModel"]},
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        dtype="float16",
    )


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()
    if args.push:
        push_gpt_hf(
            cfg_path=os.path.join(root_dir, "checkpoints/config.yaml"),
            model_dir=os.path.join(root_dir, "checkpoints"),
        )
    if args.load:
        load_gpt_hf(
            repo_id="janak22/index-tts-gpt",
        )
