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
from vllm.model_executor.models.gpt2 import GPT2Model
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from huggingface_hub import snapshot_download


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

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.get_input_embeddings(input_ids)

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

        if inputs_embeds is not None and inputs_embeds.shape[0] != 1:
            mel_len = inputs_embeds.shape[0]  # target_len
            text_inputs = input_ids[mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            mel_emb = inputs_embeds
            emb = torch.cat([mel_emb, text_emb], dim=0)
            self.cached_mel_emb = mel_emb
        elif positions.shape[0] != 1:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.emb(positions)

        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.emb(
                positions - self.cached_mel_emb.shape[1]
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
        del self.gpt
        del self.mel_pos_embedding
        del self.text_pos_embedding
        del self.mel_layer_pos_embedding
        del self.text_layer_pos_embedding

        gc.collect()
        torch.cuda.empty_cache()

        path = snapshot_download(
            "janak22/index-tts-gpt",
            ignore_patterns=["*.tflite", "*.onnx", "*.msgpack", "*.ot", "*.h5"],
        )
        self.inference_model = LLM(
            path,
            max_model_len=1024,
            skip_tokenizer_init=True,
            enable_prompt_embeds=True,
            dtype="float32" if not half else "float16",
            gpu_memory_utilization=0.6,
        )

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(
        self,
        speech_conditioning_inputs,  # [1, 34, 1280]
        first_inputs,  # [1, L+2, 1280]
        first_head,
        second_inputs=None,  # [1, L'+2, 1280]
        second_head=None,
        get_attns=False,
        return_latent=False,
    ):
        if second_inputs is not None:
            emb = torch.cat(
                [speech_conditioning_inputs, first_inputs, second_inputs], dim=1
            )  # [1, 34+L+2+L'+2, 1280]
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(
            inputs_embeds=emb, return_dict=True, output_attentions=get_attns
        )
        if get_attns:
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]  # 34
        enc = gpt_out.last_hidden_state[
            :, offset:
        ]  # [1, L+2+L'+2 +34, 1280] -> [1, L+2+L'+2, 1280]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, : first_inputs.shape[1]], enc[
                :, -second_inputs.shape[1] :
            ]  # [1, L+2, 1280], [1, L'+2, 1280]

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def forward(
        self,
        speech_conditioning_latent,  # [1, 32, 1280]
        text_inputs,  # [1, L]
        text_lengths,
        mel_codes,  # [1, L']
        mel_codes_lengths,
        emo_speech_conditioning_latent,
        cond_mel_lengths=None,
        emo_cond_mel_lengths=None,
        emo_vec=None,
        use_speed=None,
        do_spk_cond=False,
    ):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        speech_conditioning_input: MEL float tensor, (b,1024)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(
                speech_conditioning_latent.transpose(1, 2), cond_mel_lengths
            )
        else:
            speech_conditioning_latent = speech_conditioning_latent

        if emo_vec is None:
            emo_vec_syn_ori = self.get_emo_conditioning(
                emo_speech_conditioning_latent.transpose(1, 2), emo_cond_mel_lengths
            )
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
            emo_vec = self.emo_layer(emo_vec_syn)

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)  # [1,L+1]

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)  # [1, L'+1]

        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        conds = torch.cat(
            (
                speech_conditioning_latent + emo_vec.unsqueeze(1),
                duration_emb_half.unsqueeze(1),
                duration_emb.unsqueeze(1),
            ),
            1,
        )  # [1, 34, 1280]
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )  # [1, L+2], [1, L+2]
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(
            text_inputs
        )  # [1, L+2, 1280]
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(
            mel_codes, self.start_mel_token, self.stop_mel_token
        )  # [1, L'+2], [1, L'+2]

        mel_emb = self.mel_embedding(mel_codes)  # []
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits = self.get_logits(
            conds,
            text_emb,
            self.text_head,
            mel_emb,
            self.mel_head,
            get_attns=False,
            return_latent=True,
        )  # [1, L+2, 1280], [1, L'+2, 1280]
        return mel_logits[
            :, :-2
        ]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

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
        self.inference_model.store_mel_emb(inputs_embeds)
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
            (trunc_index + self.max_mel_tokens - 1)
            if max_generate_length is None
            else trunc_index + max_generate_length
        )

        prompt_token_ids = [
            dict(prompt_token_ids=p, prompt_embeds=inputs_embeds) for p in input_ids
        ]

        output = self.inference_model.generate(
            prompt_token_ids,
            sampling_params=SamplingParams(
                temperature=hf_generate_kwargs.get("temperature", 0.8),
                max_tokens=max_length,
                top_p=hf_generate_kwargs.get("top_p", 0.8),
                top_k=hf_generate_kwargs.get("top_k", 30),
                repetition_penalty=hf_generate_kwargs.get("repetition_penalty", 10.0),
            ),
        )
        if isinstance(output, torch.Tensor):
            return output[:, trunc_index:], speech_conditioning_latent
        # GenerateOutput
        output.sequences = output.sequences[:, trunc_index:]
        return output, speech_conditioning_latent
