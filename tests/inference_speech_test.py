import time
import torch
import torchaudio
import random


def main():
    start_time = time.perf_counter()

    emo_vector = list(emo_dict.values())

    emo_audio_prompt = spk_audio_prompt
    # must always use alpha=1.0 when we don't have an external reference voice
    emo_alpha = 1.0

    audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

    inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
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
    style = self.campplus_model(feat.unsqueeze(0))  # 参考音频的全局style2[1,192]

    prompt_condition = self.s2mel.models["length_regulator"](
        S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
    )[0]  # [1, T', 512]

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

        emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
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
    segments_count = len(segments)

    text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)

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

    for seg_idx, sent in enumerate(segments):
        text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
        text_tokens = torch.tensor(
            text_tokens, dtype=torch.int32, device=self.device
        ).unsqueeze(0)  # [1, T]

        m_start_time = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast(
                text_tokens.device.type,
                enabled=self.dtype is not None,
                dtype=self.dtype,
            ):
                weight_vector_sum = torch.sum(weight_vector)
                if weight_vector_sum != 1.0 or emo_vector is None:
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
                else:
                    emovec = emovec_mat

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


if __name__ == "__main__":
    main()
