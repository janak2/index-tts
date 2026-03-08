from indextts.infer_v2 import IndexTTS2, find_most_similar_cosine
import os
import time
import random
import warnings
import torch
import torchaudio

from omegaconf import OmegaConf

from indextts.gpt.model_v2_vllm import UnifiedVoiceVLLM
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

from huggingface_hub import hf_hub_download
import safetensors
from transformers import SeamlessM4TFeatureExtractor
from subprocess import CalledProcessError

import gc


class IndexTTS2VLLM(IndexTTS2):
    def __init__(
        self,
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=False,
        use_int8=False,
        device=None,
        use_cuda_kernel=None,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            use_fp16 (bool): whether to use fp16.
            use_int8 (bool): whether to use int8.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
            use_deepspeed (bool): whether to use DeepSpeed or not.
            use_accel (bool): whether to use acceleration engine for GPT2 or not.
            use_torch_compile (bool): whether to use torch.compile for optimization or not.
        """
        torch.cuda.memory._record_memory_history(max_entries=100000)

        if device is not None:
            self.device = device
            self.use_fp16 = False if device == "cpu" else use_fp16
            self.use_cuda_kernel = (
                use_cuda_kernel is not None
                and use_cuda_kernel
                and device.startswith("cuda")
            )
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = "xpu"
            self.use_fp16 = use_fp16
            self.use_cuda_kernel = False
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.use_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.use_fp16 = False
            self.use_cuda_kernel = False
            print(">> Be patient, it may take a while to run in CPU mode.")

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.use_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token
        self.use_accel = use_accel
        self.use_torch_compile = use_torch_compile
        self.use_int8 = use_int8

        def _gpu_mem_gb():
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024**3)
            return 0.0

        print(f">> [GPU mem] before UnifiedVoiceVLLM load: {_gpu_mem_gb():.2f} GB")
        self.gpt = UnifiedVoiceVLLM(**self.cfg.gpt, use_accel=self.use_accel)
        print(f">> [GPU mem] after UnifiedVoiceVLLM load: {_gpu_mem_gb():.2f} GB")
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        print(f">> [GPU mem] before load_checkpoint: {_gpu_mem_gb():.2f} GB")
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        print(">> GPT weights restored from:", self.gpt_path)
        gpt_size_gb = sum(
            p.numel() * p.element_size() for p in self.gpt.parameters()
        ) / (1024**3)

        for name, module in self.gpt.named_children():
            mem_gb = sum(p.numel() * p.element_size() for p in module.parameters()) / (
                1024**3
            )
            print(f">> [GPT member] {name}: {mem_gb:.4f} GB")

        del self.gpt.gpt
        del self.gpt.mel_layer_pos_embedding
        del self.gpt.text_layer_pos_embedding

        gc.collect()
        torch.cuda.empty_cache()

        print(f">> GPT size: {gpt_size_gb:.2f} GB")
        print(f">> [GPU mem] after GPT load: {_gpu_mem_gb():.2f} GB")

        if use_deepspeed:
            try:
                import deepspeed
            except (ImportError, OSError, CalledProcessError) as e:
                use_deepspeed = False
                print(
                    f">> Failed to load DeepSpeed. Falling back to normal inference. Error: {e}"
                )

        print(f">> [GPU mem] before post_init_gpt2_config: {_gpu_mem_gb():.2f} GB")
        self.gpt.post_init_gpt2_config(
            use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16
        )
        print(f">> [GPU mem] after post_init_gpt2_config: {_gpu_mem_gb():.2f} GB")

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import (
                    activation1d,
                )

                print(
                    ">> Preload custom CUDA kernel for BigVGAN",
                    activation1d.anti_alias_activation_cuda,
                )
            except Exception as e:
                print(
                    ">> Failed to load custom CUDA kernel for BigVGAN. Falling back to torch."
                )
                print(f"{e!r}")
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        print(f">> [GPU mem] before semantic_model load: {_gpu_mem_gb():.2f} GB")
        self.semantic_model, self.semantic_mean, self.semantic_std = (
            build_semantic_model(
                os.path.join(self.model_dir, self.cfg.w2v_stat), use_int8=self.use_int8
            )
        )

        # int8 model does not need to be moved to device
        if not self.use_int8:
            self.semantic_model = self.semantic_model.to(self.device)
            self.semantic_model.eval()

        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        if use_fp16:
            if not self.use_int8:
                self.semantic_model = self.semantic_model.half()
            self.semantic_mean = self.semantic_mean.half()
            self.semantic_std = self.semantic_std.half()

        semantic_model_size_gb = sum(
            p.numel() * p.element_size() for p in self.semantic_model.parameters()
        ) / (1024**3)
        print(f">> semantic_model size: {semantic_model_size_gb:.2f} GB")
        print(f">> [GPU mem] after semantic_model load: {_gpu_mem_gb():.2f} GB")

        print(f">> [GPU mem] before semantic_codec load: {_gpu_mem_gb():.2f} GB")
        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download(
            "amphion/MaskGCT", filename="semantic_codec/model.safetensors"
        )
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        print(">> semantic_codec weights restored from: {}".format(semantic_code_ckpt))

        semantic_codec_size_gb = sum(
            p.numel() * p.element_size() for p in self.semantic_codec.parameters()
        ) / (1024**3)
        print(f">> semantic_codec size: {semantic_codec_size_gb:.2f} GB")
        print(f">> [GPU mem] after semantic_codec load: {_gpu_mem_gb():.2f} GB")

        print(f">> [GPU mem] before s2mel load: {_gpu_mem_gb():.2f} GB")
        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models["cfm"].estimator.setup_caches(
            max_batch_size=1, max_seq_length=8192
        )
        s2mel_size_gb = sum(
            p.numel() * p.element_size() for p in self.s2mel.parameters()
        ) / (1024**3)
        print(f">> s2mel size: {s2mel_size_gb:.2f} GB")
        print(f">> [GPU mem] after s2mel load: {_gpu_mem_gb():.2f} GB")

        # Enable torch.compile optimization if requested
        if self.use_torch_compile:
            print(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")

        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)

        # load campplus_model
        print(f">> [GPU mem] before campplus_model load: {_gpu_mem_gb():.2f} GB")
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(
            torch.load(campplus_ckpt_path, map_location="cpu")
        )
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        print(">> campplus_model weights restored from:", campplus_ckpt_path)

        campplus_model_size_gb = sum(
            p.numel() * p.element_size() for p in self.campplus_model.parameters()
        ) / (1024**3)
        print(f">> campplus_model size: {campplus_model_size_gb:.2f} GB")
        print(f">> [GPU mem] after campplus_model load: {_gpu_mem_gb():.2f} GB")

        print(f">> [GPU mem] before bigvgan load: {_gpu_mem_gb():.2f} GB")
        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(
            bigvgan_name, use_cuda_kernel=self.use_cuda_kernel
        )
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()

        self.bigvgan.eval()
        print(">> bigvgan weights restored from:", bigvgan_name)

        bigvgan_size_gb = sum(
            p.numel() * p.element_size() for p in self.bigvgan.parameters()
        ) / (1024**3)
        print(f">> bigvgan size: {bigvgan_size_gb:.2f} GB")
        print(f">> [GPU mem] after bigvgan load: {_gpu_mem_gb():.2f} GB")

        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        print(">> TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        print(">> bpe model loaded from:", self.bpe_path)

        # 加载术语词汇表（如果存在）
        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        if os.path.exists(self.glossary_path):
            self.normalizer.load_glossary_from_yaml(self.glossary_path)
            print(">> Glossary loaded from:", self.glossary_path)

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": self.cfg.s2mel["preprocess_params"]["spect_params"][
                "win_length"
            ],
            "hop_size": self.cfg.s2mel["preprocess_params"]["spect_params"][
                "hop_length"
            ],
            "num_mels": self.cfg.s2mel["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmin", 0),
            "fmax": None
            if self.cfg.s2mel["preprocess_params"]["spect_params"].get("fmax", "None")
            == "None"
            else 8000,
            "center": False,
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # 缓存参考音频：
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

        # 进度引用显示（可选）
        self.gr_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None

        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    def infer_vllm(
        self,
        spk_audio_prompt,
        text,
        output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        verbose=False,
        max_text_tokens_per_segment=120,
        stream_return=False,
        more_segment_before=0,
        **generation_kwargs,
    ):
        if stream_return:
            return self.infer_generator_vllm(
                spk_audio_prompt,
                text,
                output_path,
                emo_audio_prompt,
                emo_alpha,
                emo_vector,
                use_emo_text,
                emo_text,
                use_random,
                interval_silence,
                verbose,
                max_text_tokens_per_segment,
                stream_return,
                more_segment_before,
                **generation_kwargs,
            )
        else:
            try:
                return list(
                    self.infer_generator_vllm(
                        spk_audio_prompt,
                        text,
                        output_path,
                        emo_audio_prompt,
                        emo_alpha,
                        emo_vector,
                        use_emo_text,
                        emo_text,
                        use_random,
                        interval_silence,
                        verbose,
                        max_text_tokens_per_segment,
                        stream_return,
                        more_segment_before,
                        **generation_kwargs,
                    )
                )[0]
            except IndexError:
                return None

    def infer_generator_vllm(
        self,
        spk_audio_prompt,
        text,
        output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        verbose=False,
        max_text_tokens_per_segment=120,
        stream_return=False,
        quick_streaming_tokens=0,
        **generation_kwargs,
    ):
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(
                f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                f"emo_text:{emo_text}"
            )
        start_time = time.perf_counter()

        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            start_time = time.perf_counter()
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
            end_time = time.perf_counter()
            print(f"QwenEmotion inference time: {end_time - start_time:.2f} seconds")
            print(f"detected emotion vectors from text: {emo_dict}")
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [
                    int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector
                ]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        # 如果参考音频改变了，才需要重新生成, 提升速度
        if (
            self.cache_spk_cond is None
            or self.cache_spk_audio_prompt != spk_audio_prompt
        ):
            if self.cache_spk_cond is not None:
                self.cache_spk_cond = None
                self.cache_s2mel_style = None
                self.cache_s2mel_prompt = None
                self.cache_mel = None
                torch.cuda.empty_cache()
            audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

            inputs = self.extract_features(
                audio_16k, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(
                self.device
            )  # [1, T, 160] # T = len(audio_16k)//320 = 121
            attention_mask = attention_mask.to(self.device)  # [1, T]
            spk_cond_emb = self.get_emb(input_features, attention_mask)  # [1, T, 1024]

            _, S_ref = self.semantic_codec.quantize(
                spk_cond_emb.to(next(self.semantic_codec.parameters()).dtype)
            )  # [1, T, 1024]
            ref_mel = self.mel_fn(
                audio_22k.to(spk_cond_emb.device).float()
            )  # [1, 80, T'] # T' = len(audio_22k)//256 = 210
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            feat = torchaudio.compliance.kaldi.fbank(
                audio_16k.to(ref_mel.device),
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )  # [T*2, 80]
            feat = feat - feat.mean(
                dim=0, keepdim=True
            )  # feat2另外一个滤波器能量组特征[922, 80]
            style = self.campplus_model(
                feat.unsqueeze(0)
            )  # 参考音频的全局style2[1,192]

            prompt_condition = self.s2mel.models["length_regulator"](
                S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
            )[0]  # [1, T', 512]

            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel

        if emo_vector is not None:
            weight_vector = torch.tensor(emo_vector, device=self.device)  # [8]
            if use_random:
                random_index = [random.randint(0, x - 1) for x in self.emo_num]
            else:
                random_index = [
                    find_most_similar_cosine(style, tmp)
                    for tmp in self.spk_matrix  # 8x [n, 192]
                ]  # [8]

            emo_matrix = [
                tmp[index].unsqueeze(0)
                for index, tmp in zip(random_index, self.emo_matrix)  # 8x [n, 1280]
            ]
            emo_matrix = torch.cat(emo_matrix, 0)  # [8, 1280]
            emovec_mat = weight_vector.unsqueeze(1) * emo_matrix  # [8, 1280]
            emovec_mat = torch.sum(emovec_mat, 0)  # [1280]
            emovec_mat = emovec_mat.unsqueeze(0)  # [1, 1280]

        if (
            self.cache_emo_cond is None
            or self.cache_emo_audio_prompt != emo_audio_prompt
        ):
            if self.cache_emo_cond is not None:
                self.cache_emo_cond = None
                torch.cuda.empty_cache()
            emo_audio, _ = self._load_and_cut_audio(
                emo_audio_prompt, 15, verbose, sr=16000
            )
            emo_inputs = self.extract_features(
                emo_audio, sampling_rate=16000, return_tensors="pt"
            )
            emo_input_features = emo_inputs["input_features"]
            emo_attention_mask = emo_inputs["attention_mask"]
            emo_input_features = emo_input_features.to(self.device)
            emo_attention_mask = emo_attention_mask.to(self.device)
            emo_cond_emb = self.get_emb(
                emo_input_features, emo_attention_mask
            )  # [1, T, 1024]

            self.cache_emo_cond = emo_cond_emb
            self.cache_emo_audio_prompt = emo_audio_prompt
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")
        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )
        segments_count = len(segments)

        text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        if self.tokenizer.unk_token_id in text_token_ids:
            print(
                f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):"
            )
            print(
                "     Tokens which can't be encoded: ",
                [
                    t
                    for t, id in zip(text_tokens_list, text_token_ids)
                    if id == self.tokenizer.unk_token_id
                ],
            )
            print(
                "     Consider updating the BPE model or modifying the text to avoid unknown tokens."
            )

        if verbose:
            print("text_tokens_list:", text_tokens_list)
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        sampling_rate = 22050

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        silence = None  # for stream_return
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(
                0.2 + 0.7 * seg_idx / segments_count,
                f"speech synthesis {seg_idx + 1}/{segments_count}...",
            )

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(
                text_tokens, dtype=torch.int32, device=self.device
            ).unsqueeze(0)  # [1, T]
            if verbose:
                print(text_tokens)
                print(
                    f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}"
                )
                # debug tokenizer
                text_token_syms = self.tokenizer.convert_ids_to_tokens(
                    text_tokens[0].tolist()
                )
                print(
                    "text_token_syms is same as segment tokens", text_token_syms == sent
                )

            m_start_time = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(
                    text_tokens.device.type,
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor(
                            [spk_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        torch.tensor(
                            [emo_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        alpha=emo_alpha,
                    )  # [1, 1280]

                    if emo_vector is not None:
                        emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
                        # emovec = emovec_mat

                    codes, speech_conditioning_latent = self.gpt.inference_speech(
                        spk_cond_emb,
                        text_tokens,
                        emo_cond_emb,
                        cond_lengths=torch.tensor(
                            [spk_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        emo_cond_lengths=torch.tensor(
                            [emo_cond_emb.shape[-1]], device=text_tokens.device
                        ),
                        emo_vec=emovec,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=autoregressive_batch_size,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        **generation_kwargs,
                    )  # [1, L'], [1, 32, 1280]

                gpt_gen_time += time.perf_counter() - m_start_time
                if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                    warnings.warn(
                        f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                        f"Input text tokens: {text_tokens.shape[1]}. "
                        f"Consider reducing `max_text_tokens_per_segment`({max_text_tokens_per_segment}) or increasing `max_mel_tokens`.",
                        category=RuntimeWarning,
                    )
                    has_warned = True

                code_lens = torch.tensor(
                    [codes.shape[-1]], device=codes.device, dtype=codes.dtype
                )
                #                 if verbose:
                #                     print(codes, type(codes))
                #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
                #                     print(f"code len: {code_lens}")

                code_lens = []
                max_code_len = 0
                for code in codes:
                    if self.stop_mel_token not in code:
                        code_len = len(code)
                    else:
                        len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                        code_len = len_[0].item() if len_.numel() > 0 else len(code)
                    code_lens.append(code_len)
                    max_code_len = max(max_code_len, code_len)
                codes = codes[:, :max_code_len]
                code_lens = torch.LongTensor(code_lens)
                code_lens = code_lens.to(self.device)
                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                m_start_time = time.perf_counter()
                use_speed = (
                    torch.zeros(spk_cond_emb.size(0)).to(spk_cond_emb.device).long()
                )
                # with torch.amp.autocast(
                #     text_tokens.device.type,
                #     enabled=self.dtype is not None,
                #     dtype=self.dtype,
                # ):
                #     latent = self.gpt(
                #         speech_conditioning_latent,
                #         text_tokens,
                #         torch.tensor(
                #             [text_tokens.shape[-1]], device=text_tokens.device
                #         ),
                #         codes,
                #         torch.tensor([codes.shape[-1]], device=text_tokens.device),
                #         emo_cond_emb,
                #         cond_mel_lengths=torch.tensor(
                #             [spk_cond_emb.shape[-1]], device=text_tokens.device
                #         ),
                #         emo_cond_mel_lengths=torch.tensor(
                #             [emo_cond_emb.shape[-1]], device=text_tokens.device
                #         ),
                #         emo_vec=emovec,
                #         use_speed=use_speed,
                #     )  # [1, L', 1280]
                #     gpt_forward_time += time.perf_counter() - m_start_time

                dtype = None
                with torch.amp.autocast(
                    text_tokens.device.type, enabled=dtype is not None, dtype=dtype
                ):
                    m_start_time = time.perf_counter()
                    diffusion_steps = 25
                    inference_cfg_rate = 0.7
                    # latent = self.s2mel.models["gpt_layer"](
                    #     latent
                    # )  # [1, L', 1280] -> [1, L', 1024]
                    S_infer = self.semantic_codec.quantizer.vq2emb(
                        codes.unsqueeze(1)
                    )  # [1, 1024, L']
                    S_infer = S_infer.transpose(1, 2)  # [1, L', 1024]
                    latent = torch.zeros_like(S_infer)
                    S_infer = S_infer + latent  # [1, L', 1024]
                    target_lengths = (code_lens * 1.72).long()

                    cond = self.s2mel.models["length_regulator"](
                        S_infer, ylens=target_lengths, n_quantizers=3, f0=None
                    )[0]  # [1, 1.72*L', 512]
                    cat_condition = torch.cat(
                        [prompt_condition, cond], dim=1
                    )  # [1, 1.72*L' + T', 512]
                    vc_target = self.s2mel.models["cfm"].inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                        ref_mel,
                        style,
                        None,
                        diffusion_steps,
                        inference_cfg_rate=inference_cfg_rate,
                    )  # [1, 80, 1.72*L' + T']
                    vc_target = vc_target[:, :, ref_mel.size(-1) :]  # [1, 80, 1.72*L']
                    s2mel_time += time.perf_counter() - m_start_time

                    m_start_time = time.perf_counter()
                    wav = (
                        self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                    )  # [1, L'']
                    print(wav.shape)
                    bigvgan_time += time.perf_counter() - m_start_time
                    wav = wav.squeeze(1)

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
                if verbose:
                    print(
                        f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max()
                    )
                # wavs.append(wav[:, :-512])
                wavs.append(wav.cpu())  # to cpu before saving
                if stream_return:
                    yield wav.cpu()
                    if silence == None:
                        silence = self.interval_silence(
                            wavs,
                            sampling_rate=sampling_rate,
                            interval_silence=interval_silence,
                        )
                    yield silence
        end_time = time.perf_counter()

        self._set_gr_progress(0.9, "saving audio...")
        wavs = self.insert_interval_silence(
            wavs, sampling_rate=sampling_rate, interval_silence=interval_silence
        )
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            if stream_return:
                return None
            yield output_path
        else:
            if stream_return:
                return None
            # 返回以符合Gradio的格式要求
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            yield (sampling_rate, wav_data)


if __name__ == "__main__":
    import torch

    torch.cuda.reset_peak_memory_stats()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    prompt_wav = os.path.join(root_dir, "examples/voice_01.wav")
    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    tts = IndexTTS2VLLM(
        cfg_path=os.path.join(root_dir, "checkpoints/config.yaml"),
        model_dir=os.path.join(root_dir, "checkpoints"),
        use_cuda_kernel=False,
        use_torch_compile=False,
        use_fp16=True,
        use_int8=True,
    )

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory used: {peak_mb:.0f} MB")

    tts.infer_vllm(
        spk_audio_prompt=prompt_wav,
        text=text,
        output_path="gen.wav",
        verbose=True,
        use_emo_text=False,
    )
    # char_size = 5
    # import string

    # time_buckets = []
    # for i in range(10):
    #     text = "".join(random.choices(string.ascii_letters, k=char_size))
    #     start_time = time.time()
    #     tts.infer(
    #         spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True
    #     )
    #     time_buckets.append(time.time() - start_time)
    # print(time_buckets)
