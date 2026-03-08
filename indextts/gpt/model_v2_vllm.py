import os

import functools
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import GPT2Config, LogitsProcessorList
from indextts.gpt.model_v2 import (
    UnifiedVoice,
    LearnedPositionEmbeddings,
    ConditioningEncoder,
    MelEncoder,
)


from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.typical_sampling import TypicalLogitsWarper
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from typing import Optional, Union, Iterable
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm.model_executor.models.gpt2 import GPT2Model
from huggingface_hub import snapshot_download

from indextts.utils.tensor_logger import save_tensor


def null_position_embeddings(range, dim, dtype=torch.float32):
    if range.ndim == 1:
        return torch.zeros((range.shape[0], dim), device=range.device, dtype=dtype)
    return torch.zeros(
        (range.shape[0], range.shape[1], dim), device=range.device, dtype=dtype
    )


class GPT2LMHeadModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.transformer = GPT2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "transformer")
        )
        del self.transformer.wpe
        self.transformer.wpe = functools.partial(
            null_position_embeddings,
            dim=self.config.n_embd,
            dtype=vllm_config.model_config.dtype,
        )
        self.final_norm = nn.LayerNorm(self.config.n_embd)
        self.mel_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

        self.lm_head = nn.Sequential(self.final_norm, self.mel_head)

        self.make_empty_intermediate_tensors = (
            self.transformer.make_empty_intermediate_tensors
        )
        self.text_pos_embedding = LearnedPositionEmbeddings(
            self.config.max_mel_seq_len, self.config.n_embd
        )
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.cached_mel_emb = torch.zeros([0, 0])
        self.input_count = 0
        self.warmup = False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        print(
            "inputs_embeds.shape",
            inputs_embeds.shape if inputs_embeds is not None else "None",
        )
        print("positions.shape", positions.shape if positions is not None else "None")
        print("input_ids.shape", input_ids.shape if input_ids is not None else "None")
        print("cached_mel_emb.shape", self.cached_mel_emb.shape if self.cached_mel_emb is not None else "None")

        # warmup or cuda graph capture
        if inputs_embeds is None:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.emb(
                positions
            )
        # prefill
        elif inputs_embeds.shape[0] != 1:
            self.cached_mel_emb = inputs_embeds
            emb = inputs_embeds

        # decode
        else:
            emb = inputs_embeds + self.text_pos_embedding.emb(
                positions - self.cached_mel_emb.shape[0] + 2
            )

        hidden_states = self.transformer(
            input_ids, positions, intermediate_tensors, inputs_embeds=emb
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        if sampling_metadata.selected_token_indices is not None:
            hidden_states = hidden_states.index_select(
                0, sampling_metadata.selected_token_indices
            )
        logits = self.lm_head(hidden_states)

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        outer_modules = (
            "text_pos_embedding.",
            "embeddings.",
            "final_norm.",
            "mel_head.",
        )

        def _remap_weights(weights):
            for name, tensor in weights:
                if (
                    not name.startswith("transformer.")
                    and not name.startswith("lm_head")
                    and not any(name.startswith(m) for m in outer_modules)
                ):
                    name = "transformer." + name
                yield name, tensor

        return loader.load_weights(_remap_weights(weights))


ModelRegistry.register_model("GPT2InferenceModel", GPT2LMHeadModel)


class UnifiedVoiceVLLM(UnifiedVoice):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            condition_type: perceiver, gst or default encoder
        """
        super().__init__(*args, **kwargs)

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        path = snapshot_download(
            "janak22/index-tts-gpt",
            ignore_patterns=["*.tflite", "*.onnx", "*.msgpack", "*.ot", "*.h5"],
        )
        self.inference_model = LLM(
            path,
            max_model_len=self.max_mel_tokens + 2 + self.max_conditioning_inputs,
            max_num_seqs=8,
            skip_tokenizer_init=True,
            enable_prompt_embeds=True,
            dtype="float32" if not half else "float16",
            gpu_memory_utilization=0.35,
            # enforce_eager=True,
        )


    def inference_speech(
        self,
        speech_condition,  # [1,T,1024]
        text_inputs,  # [1,L]
        emo_speech_condition=None,  # [1,T,1024]
        cond_lengths=None,  # [1]
        emo_cond_lengths=None,  # [1]
        emo_vec=None,  # [1, 1280]
        use_speed=False,
        input_tokens=None,
        num_return_sequences=1,
        max_generate_length=None,
        typical_sampling=False,
        typical_mass=0.9,
        **hf_generate_kwargs,
    ):
        """
        Args:
            speech_condition: (b, d, frames) or (d, frames)
            text_inputs: (b, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
            input_tokens: additional tokens for generation in shape (b, s) or (s,)
            max_generate_length: limit the number of generated tokens
            hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`
        """

        if speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
        if cond_lengths is None:
            cond_lengths = torch.tensor(
                [speech_condition.shape[-1]], device=speech_condition.device
            )
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor(
                [emo_speech_condition.shape[-1]], device=speech_condition.device
            )

        speech_conditioning_latent = self.get_conditioning(
            speech_condition.transpose(1, 2), cond_lengths
        )  # [1, 32, 1280]
        if emo_vec is None:
            print("compute emo vec")
            emo_vec = self.get_emo_conditioning(
                emo_speech_condition.transpose(1, 2), emo_cond_lengths
            )
            emo_vec = self.emovec_layer(emo_vec)
            emo_vec = self.emo_layer(emo_vec)
        else:
            print("Use the specified emotion vector")

        tmp = torch.zeros(text_inputs.size(0)).to(text_inputs.device)  # [1]
        duration_emb = self.speed_emb(torch.zeros_like(tmp).long())  # [1, 1280]
        duration_emb_half = self.speed_emb(torch.ones_like(tmp).long())  # [1, 1280]
        conds_latent = torch.cat(
            (
                speech_conditioning_latent + emo_vec.unsqueeze(1),  # [1, 32, 1280]
                duration_emb_half.unsqueeze(1),  # [1, 1, 1280]
                duration_emb.unsqueeze(1),  # [1, 1, 1280]
            ),
            1,
        )  # [1, 34, 1280]
        input_ids, inputs_embeds, attention_mask = self.prepare_gpt_inputs(
            conds_latent, text_inputs
        )  # [b, target_len+1], [b, target_len, 1280], [b, target_len+1] # target_len = 34 + L + 2
        if input_tokens is None:
            inputs = input_ids
        else:
            if input_tokens.ndim == 1:
                input_tokens = input_tokens.unsqueeze(0)
            assert num_return_sequences % input_tokens.shape[0] == 0, (
                "The num_return_sequences must be divisible by the batch number of input_tokens"
            )
            assert num_return_sequences % text_inputs.shape[0] == 0, (
                "The num_return_sequences must be divisible by the batch number of text_inputs"
            )
            b = num_return_sequences // input_ids.shape[0]
            if b > 1:
                input_ids = input_ids.repeat(b, 1)
                attention_mask = attention_mask.repeat(b, 1)
            input_tokens = input_tokens.repeat(
                num_return_sequences // input_tokens.shape[0], 1
            )
            inputs = torch.cat([input_ids, input_tokens], dim=1)
            attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)
        trunc_index = inputs.shape[1]
        logits_processor = LogitsProcessorList()
        if typical_sampling:
            # employ custom typical sampling
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(
                    f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}"
                )
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(
                TypicalLogitsWarper(
                    mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep
                )
            )
        max_length = (
            (self.max_mel_tokens + 2 + self.max_conditioning_inputs - trunc_index - 1)
            if max_generate_length is None
            else max_generate_length
        )

        prompt_token_ids = [
            dict(prompt_token_ids=p.tolist(), prompt_embeds=inputs_embeds)
            for p in input_ids
        ]

        print("Starting generation")
        output = self.inference_model.generate(
            prompt_token_ids,
            sampling_params=SamplingParams(
                temperature=hf_generate_kwargs["temperature"]
                if hf_generate_kwargs.get("do_sample", True)
                else 0,
                max_tokens=max_length,
                best_of=hf_generate_kwargs["num_beams"],
                top_p=hf_generate_kwargs["top_p"],
                top_k=hf_generate_kwargs["top_k"],
                repetition_penalty=hf_generate_kwargs["repetition_penalty"],
                stop_token_ids=[self.stop_mel_token],
            ),
        )
        tokens = []
        for o in output:
            tokens.append(o.outputs[0].token_ids)
        tokens = torch.tensor(tokens, device=speech_conditioning_latent.device)
        return tokens, speech_conditioning_latent

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,  # [1, 34, 1280]
        text_inputs: torch.Tensor,  # [1, L]
    ):
        """
        Prepare the inputs for the GPT2InferenceModel to generate.
        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)
        Returns:
            input_ids: (b, s+1) the input ids for the GPT2InferenceModel.generate()
            inputs_embeds: (b, s+1, dim) the input embeddings for the GPT2InferenceModel.forward()
            attention_mask: (b, s+1) the attention mask for the GPT2InferenceModel.generate()
        """
        b, L = text_inputs.shape[:2]
        device = text_inputs.device
        single_cond = (
            conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        )
        if not single_cond:
            assert conditional_latents.shape[0] == b, (
                f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
            )
        batched_mel_emb = []
        attention_masks = []
        target_len = conditional_latents.shape[1] + L + 2
        for i in range(b):
            valid_mask = (text_inputs[i] != self.stop_text_token) & (
                text_inputs[i] != self.start_text_token
            )
            text_input = text_inputs[i][valid_mask]  # [L]
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)  # [L+2]
            text_input_pos = torch.arange(
                0, text_input.size(-1), device=device
            )  # [L+2]
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(
                text_input_pos
            )  # [L+2, 1280]
            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [
                conditional_latents.squeeze(0)
                if single_cond
                else conditional_latents[i],
                text_emb,
            ]
            # +1 for the start_mel_token
            attention_mask = torch.ones(target_len + 1, dtype=torch.long, device=device)
            # check this text input is padded
            padding: int = L + 2 - text_input.size(-1)
            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = torch.zeros(
                    (padding, conditional_latents.size(-1)),
                    dtype=text_emb.dtype,
                    device=device,
                )  # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb)  # [target_len, 1280]
            assert mel_emb.shape[0] == target_len, (
                f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            )
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        # [b, s, dim]
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        # [b, s+1]
        attention_mask = torch.stack(attention_masks, dim=0)
        # [b, s+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )  # [b, target_len+1]
        fake_inputs[:, -1] = self.start_mel_token

        last_token_emb = self.mel_embedding(
            torch.tensor([self.start_mel_token], device=device)
        ).unsqueeze(0)
        last_token_emb = last_token_emb + self.mel_pos_embedding(last_token_emb)
        batched_mel_emb = torch.cat(
            [batched_mel_emb, last_token_emb.repeat(b, 1, 1)], dim=1
        )

        return fake_inputs, batched_mel_emb, attention_mask
