import time
import torch
import torchaudio
import random
import os
from indextts.infer_v2_vllm import IndexTTS2VLLM

from transformers import SeamlessM4TFeatureExtractor
from indextts.utils.maskgct_utils import build_semantic_model
from indextts.utils.front import TextNormalizer, TextTokenizer
from omegaconf import OmegaConf
from indextts.gpt.model_v2 import UnifiedVoice
from indextts.gpt.model_v2_vllm import UnifiedVoiceVLLM
from indextts.utils.checkpoint import load_checkpoint

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class InferenceSpeechTest(IndexTTS2VLLM):
    def __init__(
        self,
        cfg_path,
        model_dir,
        use_int8=False,
        use_fp16=False,
        use_accel=False,
        use_deepspeed=False,
        use_vllm=False,
    ):
        self.model_dir = model_dir
        self.cfg = OmegaConf.load(cfg_path)
        self.use_int8 = use_int8
        self.use_fp16 = use_fp16
        self.use_accel = use_accel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.use_fp16 else None

        if use_vllm:
            self.gpt = UnifiedVoiceVLLM(**self.cfg.gpt, use_accel=self.use_accel)
        else:
            self.gpt = UnifiedVoice(**self.cfg.gpt, use_accel=self.use_accel)

        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.use_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()

        self.gpt.post_init_gpt2_config(
            use_deepspeed=use_deepspeed, kv_cache=True, half=self.use_fp16
        )

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
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

        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer(enable_glossary=True)
        self.normalizer.load()
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)

    def test_inference(
        self,
        spk_audio_prompt,
        text,
        max_text_tokens_per_segment=120,
        quick_streaming_tokens=0,
        **generation_kwargs,
    ):
        emo_audio_prompt = spk_audio_prompt

        emo_alpha = 1.0

        audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose=False)
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

        emo_audio, _ = self._load_and_cut_audio(
            emo_audio_prompt, 15, verbose=False, sr=16000
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

        text_tokens_list = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_segments(
            text_tokens_list,
            max_text_tokens_per_segment,
            quick_streaming_tokens=quick_streaming_tokens,
        )

        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)

        for seg_idx, sent in enumerate(segments):
            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(
                text_tokens, dtype=torch.int32, device=self.device
            ).unsqueeze(0)  # [1, T]

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

                    return codes, speech_conditioning_latent


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    prompt_wav = os.path.join(root_dir, "examples/voice_01.wav")
    text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    tts = InferenceSpeechTest(
        cfg_path=os.path.join(root_dir, "checkpoints/config.yaml"),
        model_dir=os.path.join(root_dir, "checkpoints"),
        use_fp16=True,
        use_int8=True,
    )

    codes, speech_conditioning_latent = tts.test_inference(
        spk_audio_prompt=prompt_wav,
        text=text,
    )
    print(codes.shape)
    print(speech_conditioning_latent.shape)
