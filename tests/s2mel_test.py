import os

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
import time
import warnings

import torch
import torchaudio

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
from pathlib import Path

import safetensors
import soundfile as sf
from huggingface_hub import hf_hub_download
from indextts.infer_v2 import IndexTTS2
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel, load_checkpoint2
from indextts.utils.maskgct_utils import build_semantic_codec
from omegaconf import OmegaConf


class S2MelTest(IndexTTS2):
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

        # Enable torch.compile optimization if requested
        if self.use_torch_compile:
            print(">> Enabling torch.compile optimization")
            self.s2mel.enable_torch_compile()
            print(">> torch.compile optimization enabled successfully")

        self.s2mel.eval()
        print(">> s2mel weights restored from:", s2mel_path)

        # load campplus_model
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

    @torch.no_grad()
    def test_s2mel(
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

        tensor_dir = Path(__file__).parent.parent / "tensors"

        latent = torch.load(tensor_dir / "latent.pt")
        codes = torch.load(tensor_dir / "codes.pt")
        code_lens = torch.load(tensor_dir / "code_lens.pt")
        prompt_condition = torch.load(tensor_dir / "prompt_condition.pt")
        style = torch.load(tensor_dir / "style.pt")
        ref_mel = torch.load(tensor_dir / "ref_mel.pt")

        dtype = None
        s2mel_time = 0
        bigvgan_time = 0
        start_time = time.perf_counter()
        wavs = []
        silence = None
        sampling_rate = 22050
        with torch.amp.autocast(
            latent.device.type, enabled=dtype is not None, dtype=dtype
        ):
            m_start_time = time.perf_counter()
            diffusion_steps = 25
            inference_cfg_rate = 0.7
            latent = self.s2mel.models["gpt_layer"](
                latent
            )  # [1, L', 1280] -> [1, L', 1024]
            S_infer = self.semantic_codec.quantizer.vq2emb(
                codes.unsqueeze(1)
            )  # [1, 1024, L']
            S_infer = S_infer.transpose(1, 2)  # [1, L', 1024]
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

            vc_target_old = torch.load(tensor_dir / "vc_target.pt")
            print(vc_target_old.shape)
            print(vc_target.shape)
            print(torch.allclose(vc_target_old, vc_target))
            print(vc_target_old.min(), vc_target.min())
            print(vc_target_old.max(), vc_target.max())
            print(vc_target_old.mean(), vc_target.mean())
            print(vc_target_old.std(), vc_target.std())
            print(vc_target_old.var(), vc_target.var())

            m_start_time = time.perf_counter()
            wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)  # [1, L'']
            print(wav.shape)
            bigvgan_time += time.perf_counter() - m_start_time
            wav = wav.squeeze(1)

        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
        if verbose:
            print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
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

    @torch.no_grad()
    def test_bigvgan(
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

        tensor_dir = Path(__file__).parent.parent / "tensors"

        latent = torch.load(tensor_dir / "latent.pt")
        codes = torch.load(tensor_dir / "codes.pt")
        code_lens = torch.load(tensor_dir / "code_lens.pt")
        prompt_condition = torch.load(tensor_dir / "prompt_condition.pt")
        style = torch.load(tensor_dir / "style.pt")
        ref_mel = torch.load(tensor_dir / "ref_mel.pt")
        vc_target = torch.load(tensor_dir / "vc_target.pt")

        dtype = None
        s2mel_time = 0
        bigvgan_time = 0
        start_time = time.perf_counter()
        wavs = []
        silence = None
        sampling_rate = 22050
        wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)  # [1, L'']
        print(wav.shape)
        wav = wav.squeeze(1)

        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
        if verbose:
            print(f"wav shape: {wav.shape}", "min:", wav.min(), "max:", wav.max())
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
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # save audio
        wav = wav.cpu()  # to cpu
        wav = wav.to(torch.float32) / wav.abs().max()
        if output_path:
            # 直接保存音频到指定路径中
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav, sampling_rate)
            sf.write("sf" + output_path, wav.cpu().numpy().squeeze(), sampling_rate)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="s2mel")
    args = parser.parse_args()

    import torch

    torch.backends.cudnn.enabled = False
    # or
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.cuda.reset_peak_memory_stats()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    prompt_wav = os.path.join(root_dir, "examples/voice_01.wav")
    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    tts = S2MelTest(
        cfg_path=os.path.join(root_dir, "checkpoints/config.yaml"),
        model_dir=os.path.join(root_dir, "checkpoints"),
        use_cuda_kernel=False,
        use_torch_compile=False,
        use_fp16=True,
        use_int8=True,
    )

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory used: {peak_mb:.0f} MB")

    if args.mode == "s2mel":
        for _ in tts.test_s2mel(
            spk_audio_prompt=prompt_wav,
            text=text,
            output_path="test.wav",
            verbose=True,
        ):
            pass
    elif args.mode == "bigvgan":
        for _ in tts.test_bigvgan(
            spk_audio_prompt=prompt_wav,
            text=text,
            output_path="test.wav",
            verbose=True,
        ):
            pass
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    print("Test completed")
