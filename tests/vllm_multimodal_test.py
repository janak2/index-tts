import time
from collections.abc import Mapping, Sequence
from random import randint, seed
from typing import Mapping, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import BatchFeature

# from nanovllm import LLM, SamplingParams
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.config import VllmConfig
from vllm.model_executor.models.gpt2 import GPT2LMHeadModel
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder


class GPT2ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}


class GPT2DummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        return "<image>" * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        target_width, target_height = (224, 224)

        return {"image": torch.randn(1, 3, target_width, target_height)}


class GPT2MultiModalProcessor(BaseMultiModalProcessor):
    # Copied from BaseMultiModalProcessor
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    GPT2MultiModalProcessor,
    info=GPT2ProcessingInfo,
    dummy_inputs=GPT2DummyInputsBuilder,
)
class GPT2TestModel(GPT2LMHeadModel, SupportsMultiModal):
    supports_multimodal_raw_input_only = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def generate(self, pixel_values: torch.Tensor, *args, **kwargs):
        print(pixel_values.shape)
        return super().generate(*args, **kwargs)

    # @classmethod
    # def get_placeholder_str(cls, modality: str, i: int) -> str | None:
    #     if modality.startswith("image"):
    #         return "<image>"

    #     raise ValueError("Only image modality is supported")


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
        skip_tokenizer_init=False,
        dtype="float16",
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
    prompt_token_ids = [
        dict(prompt_token_ids=p, pixel_values=torch.randn(1, 3, 224, 224))
        for p in prompt_token_ids
    ]

    t = time.time()
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t
    print(f"Time: {t:.2f}s")
    print(len(outputs[0].outputs[0].token_ids))


if __name__ == "__main__":
    main()
