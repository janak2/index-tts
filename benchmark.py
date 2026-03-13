from indextts.infer_v2 import IndexTTS2
from indextts.infer_v2_vllm import IndexTTS2VLLM
import time
import os
import json

PROMPTS = [
    "The morning sun cast long golden shadows across the quiet village square, where a single bicycle leaned against the old stone fountain.",
    "Attention passengers: Flight seven-forty-two to Barcelona is now boarding at Gate B-twelve. Please have your boarding passes ready.",
    "Did you really think I wouldn't notice? Oh, come on — that's the third time this week you've eaten my leftover pizza!",
    # "To activate the device, press and hold the power button for three seconds, then release it when the blue indicator light begins to flash.",
    # "Once upon a time, in a kingdom buried beneath the clouds, there lived a tiny dragon who was terribly, hopelessly afraid of fire.",
]

root_dir = os.path.dirname(os.path.abspath(__file__))

prompt_wav = os.path.join(root_dir, "examples/voice_01.wav")

def benchmark_v2(use_cuda_kernel=False, use_torch_compile=False, use_accel=False, use_deepspeed=False):
    tts = IndexTTS2(
        use_cuda_kernel=use_cuda_kernel,
        use_torch_compile=use_torch_compile,
        use_accel=use_accel,
        use_deepspeed=use_deepspeed,
        use_fp16=True,
        use_int8=True,
    )
    rtf = 0

    # warm up
    for _ in range(2):
        tts.infer(
            text="What is the capital of France?",
            output_path=None,
            spk_audio_prompt=prompt_wav,
        )

    for prompt in PROMPTS:
        start_time = time.time()
        sr, wav = tts.infer(
            text=prompt,
            output_path=None,
            spk_audio_prompt=prompt_wav,
        )
        end_time = time.time()

        rtf += (end_time - start_time) / (len(wav) / sr)
    return rtf / len(PROMPTS)

def benchmark_v2_vllm(use_cuda_kernel=False, use_torch_compile=False, ):
    tts = IndexTTS2VLLM(
        use_cuda_kernel=use_cuda_kernel,
        use_torch_compile=use_torch_compile,
        use_fp16=True,
        use_int8=True,
    )

    # warm up
    for _ in range(2):
        tts.infer_vllm(
            text="What is the capital of France?",
            output_path=None,
            spk_audio_prompt=prompt_wav,
        )


    rtf = 0
    for prompt in PROMPTS:
        start_time = time.time()
        sr, wav = tts.infer_vllm(
            text=prompt,
            output_path=None,
            spk_audio_prompt=prompt_wav,
        )
        end_time = time.time()

        rtf += (end_time - start_time) / (len(wav) / sr)
    return rtf / len(PROMPTS)


if __name__ == "__main__":
    RESULTS = []
    for use_cuda_kernel, use_torch_compile in [(True, True), (False, False)]:
        for use_vllm, use_accel, use_deepspeed in [(False, False, False), (True, False, False), (False, True, False), (False, False, True)]:
            if use_vllm:
                rtf = benchmark_v2_vllm(use_cuda_kernel, use_torch_compile)
            else:
                rtf = benchmark_v2(use_cuda_kernel, use_torch_compile, use_accel, use_deepspeed)
            RESULTS.append({
                "use_vllm": use_vllm,
                "use_cuda_kernel": use_cuda_kernel,
                "use_torch_compile": use_torch_compile,
                "use_accel": use_accel,
                "use_deepspeed": use_deepspeed,
                "rtf": rtf,
            })
            print(f"V2 RTF: {rtf}, use_vllm: {use_vllm}, use_cuda_kernel: {use_cuda_kernel}, use_torch_compile: {use_torch_compile}, use_accel: {use_accel}, use_deepspeed: {use_deepspeed}")

    with open("results.json", "w") as f:
        json.dump(RESULTS, f)