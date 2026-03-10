import time
from abc import abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from random import randint
from typing import TypeVar

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import (
    BatchFeature,
    LlavaConfig,
    PixtralVisionConfig,
)
from vllm import LLM, ModelRegistry, SamplingParams
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.clip import CLIPVisionModel
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.llava import (
    BaseLlavaProcessingInfo,
    LlavaImageEmbeddingInputs,
    LlavaImageInputs,
    LlavaImagePixelInputs,
    LlavaMultiModalProjector,
    LlavaProcessingInfo,
    PixtralHFImagePixelInputs,
    PixtralHFMultiModalProcessor,
    PixtralHFProcessingInfo,
    init_vision_tower_for_llava,
)
from vllm.model_executor.models.pixtral import PixtralHFVisionModel
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import BaseMultiModalProcessorCache
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
    InputProcessingContext,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

_I = TypeVar("_I", bound=BaseLlavaProcessingInfo)


class BaseLlavaMultiModalProcessor(BaseMultiModalProcessor[_I]):
    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError

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

        return [PromptReplacement("image", [image_token_id], get_replacement)]


class LlavaMultiModalProcessor(BaseLlavaMultiModalProcessor[LlavaProcessingInfo]):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )


def _build_llava_or_pixtral_hf_processor(
    info: _I,
    dummy_inputs: BaseDummyInputsBuilder[_I],
    *,
    cache: BaseMultiModalProcessorCache | None = None,
) -> BaseMultiModalProcessor:
    if isinstance(info, PixtralHFProcessingInfo):
        return PixtralHFMultiModalProcessor(
            info,
            dummy_inputs,  # type: ignore
            cache=cache,
        )

    if isinstance(info, LlavaProcessingInfo):
        return LlavaMultiModalProcessor(
            info,
            dummy_inputs,  # type: ignore
            cache=cache,
        )

    raise NotImplementedError(type(info))


def _build_llava_or_pixtral_hf_info(
    ctx: InputProcessingContext,
) -> BaseLlavaProcessingInfo:
    hf_config = ctx.get_hf_config(LlavaConfig)

    if isinstance(hf_config.vision_config, PixtralVisionConfig):
        return PixtralHFProcessingInfo(ctx)

    return LlavaProcessingInfo(ctx)


class LlavaDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        return {"image": torch.zeros(num_images, 100, 2048)}


@MULTIMODAL_REGISTRY.register_processor(
    _build_llava_or_pixtral_hf_processor,
    info=_build_llava_or_pixtral_hf_info,
    dummy_inputs=LlavaDummyInputsBuilder,
)
class LlavaForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    merge_by_field_config = True
    supports_multimodal_raw_input_only = True  # <-- add this

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # mapping for new names in checkpoint saved after transformers v4.52
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"

        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # NOTE: These are special cases for Pixtral-12B in the HF-format
        # https://huggingface.co/mistral-community/pixtral-12b/blob/main/config.json  # noqa
        if (
            config.text_config.architectures is None
            and config.text_config.model_type == "mistral"
        ):
            config.text_config.architectures = ["MistralForCausalLM"]
        if (
            config.projector_hidden_act is None
            and config.vision_config.hidden_act == "gelu"
        ):
            config.projector_hidden_act = "gelu"

        # TODO: Optionally initializes this for supporting embeddings.
        if multimodal_config.get_limit_per_prompt("image"):
            self.vision_tower = init_vision_tower_for_llava(
                config,
                quant_config,
                require_post_norm=False,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.multi_modal_projector = LlavaMultiModalProjector(
                vision_hidden_size=config.vision_config.hidden_size,
                text_hidden_size=config.text_config.hidden_size,
                projector_hidden_act=config.projector_hidden_act,
                multimodal_projector_bias=config.multimodal_projector_bias,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "multi_modal_projector"),
            )
        else:
            self.vision_tower = None
            self.multi_modal_projector = None

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlavaImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if self.config.vision_config.model_type == "pixtral":
                return PixtralHFImagePixelInputs(
                    type="pixel_values_pixtral",
                    pixel_values=pixel_values,
                )

            expected_h = expected_w = self.config.vision_config.image_size
            return LlavaImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                resolve_bindings={"h": expected_h, "w": expected_w},
            )

        if image_embeds is not None:
            if self.config.vision_config.model_type == "pixtral":
                raise ValueError("Pixtral-HF does not support image_embeds.")

            return LlavaImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self,
        vision_tower: CLIPVisionModel | SiglipVisionModel | PixtralHFVisionModel,
        pixel_values: torch.Tensor | list[torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        return vision_tower(
            pixel_values,
            feature_select_strategy=self.config.vision_feature_select_strategy,
        )

    def _process_image_pixels(
        self,
        inputs: LlavaImagePixelInputs | PixtralHFImagePixelInputs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(
        self,
        image_input: LlavaImageInputs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_tower is not None
        image_features = self._process_image_pixels(image_input)

        if isinstance(image_features, torch.Tensor):
            return self.multi_modal_projector(image_features)

        feature_sizes = [image_feature.shape[0] for image_feature in image_features]

        image_embeds = self.multi_modal_projector(torch.cat(image_features))
        image_embeds = torch.split(image_embeds, feature_sizes)
        return image_embeds

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for LLaVA-1.5.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"USER: <image>\\nWhat's the content of the image?\\nASSISTANT:"`.

        Tokenizer outputs:
        `[1, 3148, 1001, 29901, 29871, 32000, 29871, 13, 5618, 29915, 29879,
        278, 2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        additional image tokens (denoted as `32000`), resulting in:
        `[1, 3148, 1001, 29901, 29871, 32000, ..., 32000, 29871, 13, 5618,
        29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566,
        29901]`.

        We insert 575 tokens so that including the original image token in the
        input, there are a total of 576 (24 * 24) image tokens, which
        corresponds to the number of image tokens inputted to the language
        model, i.e. the number of image tokens outputted by the visual encoder.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Position indices for the input tokens.
            intermediate_tensors: Intermediate tensors from prior forward pass.
            inputs_embeds: Optional tensor of input embeddings.

        Info:
            [`LlavaImageInputs`][vllm.model_executor.models.llava.LlavaImageInputs]
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = []
        if self.vision_tower is None and self.multi_modal_projector is None:
            skip_prefixes.extend(["vision_tower.", "multi_modal_projector."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


ModelRegistry.register_model("GPT2TestModel", LlavaForConditionalGeneration)


def main():
    num_seqs = 1
    max_input_len = 1024
    max_ouput_len = 1024

    path = snapshot_download(
        "bczhou/tiny-llava-v1-hf",
        ignore_patterns=["*.tflite", "*.onnx", "*.msgpack", "*.ot", "*.h5"],
    )
    llm = LLM(
        path,
        max_model_len=1024,
        max_num_seqs=num_seqs,
        hf_overrides={"architectures": ["GPT2TestModel"]},
        skip_tokenizer_init=False,
        dtype="float16",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        enable_mm_embeds=True,
        skip_mm_profiling=True,
    )

    image_token_id = 32000
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(100)] + [image_token_id]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)
        )
        for _ in range(num_seqs)
    ]
    prompt_token_ids = [
        dict(prompt_token_ids=p, multi_modal_data={"image": [torch.zeros(3, 2048)]})
        for p in prompt_token_ids
    ]

    t = time.time()
    outputs = llm.generate(
        prompt_token_ids,
        sampling_params,
        use_tqdm=False,
    )
    t = time.time() - t
    print(f"Time: {t:.2f}s")
    print(len(outputs[0].outputs[0].token_ids))


if __name__ == "__main__":
    main()
