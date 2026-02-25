import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import GPT2Config, LogitsProcessorList
from indextts.gpt.model_v2 import UnifiedVoice


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
from vllm.model_executor.models.gpt2 import _add_transformer_prefix, GPT2Model


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class GPT2LMHeadModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.transformer = GPT2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "transformer")
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

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.transformer(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
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


def build_hf_gpt_transformer(
    layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing
):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model

    gpt_config = GPT2Config(
        vocab_size=256,  # Unused.
        n_positions=max_mel_seq_len + max_text_seq_len,
        n_ctx=max_mel_seq_len + max_text_seq_len,
        n_embd=model_dim,
        n_layer=layers,
        n_head=heads,
        gradient_checkpointing=checkpointing,
        use_cache=not checkpointing,
    )
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
        None,
        None,
    )


class UnifiedVoiceVLLM(UnifiedVoice):
    def __init__(
        self,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_conditioning_inputs=1,
        mel_length_compression=1024,
        number_text_tokens=256,
        start_text_token=0,
        stop_text_token=1,
        number_mel_codes=8194,
        start_mel_token=8192,
        stop_mel_token=8193,
        train_solo_embeddings=False,
        use_mel_codes_as_input=True,
        checkpointing=True,
        types=1,
        condition_num_latent=32,
        condition_type="perceiver",
        condition_module=None,
        emo_condition_module=None,
        use_accel=False,
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
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim  # 1280
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        if condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(
                1024, model_dim, num_attn_heads=heads
            )
            self.perceiver_encoder = PerceiverResampler(
                model_dim, dim_context=model_dim, num_latents=self.cond_num
            )
        elif (
            condition_type == "conformer_perceiver"
            or condition_type == "conformer_encoder"
        ):
            self.conditioning_encoder = ConformerEncoder(
                input_size=1024,
                output_size=condition_module["output_size"],  # 512
                linear_units=condition_module["linear_units"],  # 2048
                attention_heads=condition_module["attention_heads"],  # 8
                num_blocks=condition_module["num_blocks"],  # 6
                input_layer=condition_module["input_layer"],  # "conv2d2"
            )
            if condition_type == "conformer_perceiver":
                self.perceiver_encoder = PerceiverResampler(
                    model_dim,  # 1280
                    dim_context=condition_module["output_size"],  # 512
                    ff_mult=condition_module["perceiver_mult"],  # 2
                    heads=condition_module["attention_heads"],  # 8
                    num_latents=self.cond_num,  # 32
                )
        else:
            self.conditioning_encoder = ConditioningEncoder(
                1024, model_dim, num_attn_heads=heads, mean=True
            )

        self.emo_conditioning_encoder = ConformerEncoder(
            input_size=1024,
            output_size=emo_condition_module["output_size"],
            linear_units=emo_condition_module["linear_units"],
            attention_heads=emo_condition_module["attention_heads"],
            num_blocks=emo_condition_module["num_blocks"],
            input_layer=emo_condition_module["input_layer"],
        )
        self.emo_perceiver_encoder = PerceiverResampler(
            1024,
            dim_context=emo_condition_module["output_size"],  # 512
            ff_mult=emo_condition_module["perceiver_mult"],  # 2
            heads=emo_condition_module["attention_heads"],  # 4
            num_latents=1,
        )

        self.text_embedding = nn.Embedding(
            self.number_text_tokens * types + 1, model_dim
        )
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)

        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
        (
            self.gpt,
            self.mel_pos_embedding,
            self.text_pos_embedding,
            self.mel_layer_pos_embedding,
            self.text_layer_pos_embedding,
        ) = build_hf_gpt_transformer(
            layers,
            model_dim,
            heads,
            self.max_mel_tokens + 2 + self.max_conditioning_inputs,
            self.max_text_tokens + 2,
            checkpointing,
        )
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(
                torch.randn(1, 1, model_dim) * 0.02, requires_grad=True
            )
            self.text_solo_embedding = nn.Parameter(
                torch.randn(1, 1, model_dim) * 0.02, requires_grad=True
            )
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=0.02)

        self.use_accel = use_accel
        self.accel_engine = None  # Will be initialized in post_init_gpt2_config

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )

        if self.use_accel and torch.cuda.is_available():
            # Check if flash attention is available
            try:
                import flash_attn
            except ImportError:
                raise ImportError(
                    "flash_attn is required for acceleration but not installed. Please install from https://github.com/Dao-AILab/flash-attention/releases/"
                )

            from indextts.accel import GPT2AccelModel, AccelInferenceEngine

            # Create accel model
            accel_gpt = GPT2AccelModel(gpt_config)
            accel_gpt.load_state_dict(self.gpt.state_dict(), strict=False)

            if half:
                accel_gpt = accel_gpt.half().cuda()
            else:
                accel_gpt = accel_gpt.cuda()
            accel_gpt.eval()

            lm_head_with_norm = nn.Sequential(self.final_norm, self.mel_head)
            self.accel_engine = AccelInferenceEngine(
                model=accel_gpt,
                lm_head=lm_head_with_norm,
                num_layers=self.layers,
                num_heads=self.heads,
                head_dim=self.model_dim // self.heads,
                block_size=256,
                num_blocks=16,  # Reduce to save memory (16*256 = 4096 tokens capacity)
                use_cuda_graph=True,
            )
            print("acceleration engine initialized")
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model,
                mp_size=1,
                replace_with_kernel_inject=True,
                dtype=torch.float16,
            )
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed

            self.ds_engine = deepspeed.init_inference(
                model=self.inference_model,
                mp_size=1,
                replace_with_kernel_inject=True,
                dtype=torch.float32,
            )
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        # self.inference_model = PrunedGPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
        self.gpt.wte = self.mel_embedding

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

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

        # Use accel engine if available (single sequence only)
        if self.accel_engine is not None and num_return_sequences == 1:
            output = self.accel_engine.generate(
                inputs,  # fake input_ids (all 1s + start_mel_token)
                max_new_tokens=max_length - trunc_index,
                attention_mask=attention_mask,
                temperature=hf_generate_kwargs.get("temperature", 1),
                stop_tokens=[self.stop_mel_token],
                tts_embeddings=inputs_embeds,  # [pad][cond][text] embeddings (87 tokens, NO start_mel_token)
                tts_mel_embedding=self.inference_model.embeddings,  # mel_embedding layer
                tts_text_pos_embedding=self.inference_model.text_pos_embedding,  # text_pos_embedding layer
            )
        else:
            output = self.inference_model.generate(
                inputs,
                bos_token_id=self.start_mel_token,
                pad_token_id=self.stop_mel_token,
                eos_token_id=self.stop_mel_token,
                attention_mask=attention_mask,
                max_length=max_length,
                logits_processor=logits_processor,
                num_return_sequences=num_return_sequences,
                **hf_generate_kwargs,
            )
        if isinstance(output, torch.Tensor):
            return output[:, trunc_index:], speech_conditioning_latent
        # GenerateOutput
        output.sequences = output.sequences[:, trunc_index:]
        return output, speech_conditioning_latent
