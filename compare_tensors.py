import torch
from argparse import ArgumentParser
from pathlib import Path


def compare_layers(path1, path2, atol=1e-6):
    sd1 = torch.load(path1, map_location="cpu")
    sd2 = torch.load(path2, map_location="cpu")

    for key in sd1:
        if key not in sd2:
            print(f"  {key}: MISSING in model2")
            continue
        t1 = sd1[key].float()
        t2 = sd2[key].float()
        print(key)
        print(t1.dtype, t1.shape)
        print(t2.dtype, t2.shape)

        if t2.ndim == 2 and t1.shape[0] == t2.shape[1]:
            t1 = t1.t()

        max_diff = (t1 - t2).abs().max().item()
        mean_diff = (t1 - t2).abs().mean().item()
        match = torch.allclose(t1, t2, atol=atol)
        print(
            f"  {key}: match={match}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
        )

    for key in sd2:
        if key not in sd1:
            print(f"  {key}: MISSING in model1")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--emb", action="store_true")
    args.add_argument("--logits", action="store_true")
    args.add_argument("--hidden_states", action="store_true")
    args.add_argument("--input", action="store_true")
    args.add_argument("--gpt2_input", action="store_true")
    args.add_argument("--pre_layer", action="store_true")
    args.add_argument("--pre_ln_f", action="store_true")
    args.add_argument("--block_input", action="store_true")
    args.add_argument("--block_output", action="store_true")
    args.add_argument("--block", action="store_true")
    args.add_argument("--count", type=int, default=0)
    args.add_argument("--layer", type=int, default=0)
    args = args.parse_args()

    tensor_dir = Path(__file__).parent / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    if args.emb:
        t1 = torch.load(tensor_dir / "run1_emb.pt", map_location="cpu").half()
        t2 = torch.load(tensor_dir / "run2_emb.pt", map_location="cpu")
        b = t1.shape[0]
        t2 = t2.repeat(b, 1, 1)
    elif args.hidden_states:
        t1 = torch.load(
            tensor_dir / f"run1_hidden_states_{args.count}.pt", map_location="cpu"
        )
        t2 = torch.load(
            tensor_dir / f"run2_hidden_states_{args.count}.pt", map_location="cpu"
        )
    elif args.logits:
        t1 = torch.load(
            tensor_dir / f"run1_logits_{args.count}.pt", map_location="cpu"
        ).half()
        t2 = torch.load(tensor_dir / f"run2_logits_{args.count}.pt", map_location="cpu")
    elif args.input:
        t1 = torch.load(
            tensor_dir / f"run1_input_{args.count}.pt", map_location="cpu"
        ).half()
        t2 = torch.load(tensor_dir / f"run2_input_{args.count}.pt", map_location="cpu")
    elif args.gpt2_input:
        t1 = torch.load(
            tensor_dir / f"run1_gpt2_input_{args.count}.pt", map_location="cpu"
        ).half()
        t2 = torch.load(
            tensor_dir / f"run2_gpt2_input_{args.count}.pt", map_location="cpu"
        )
    elif args.pre_layer:
        t1 = torch.load(
            tensor_dir / f"run1_pre_layer_{args.count}.pt", map_location="cpu"
        ).half()
        t2 = torch.load(
            tensor_dir / f"run2_pre_layer_{args.count}.pt", map_location="cpu"
        )
    elif args.pre_ln_f:
        t1 = torch.load(
            tensor_dir / f"run1_pre_ln_f_{args.count}.pt", map_location="cpu"
        ).half()
        t2 = torch.load(
            tensor_dir / f"run2_pre_ln_f_{args.count}.pt", map_location="cpu"
        )
    elif args.block_input:
        t1 = torch.load(
            tensor_dir / f"run1_block_input_{args.layer}_{args.count}.pt",
            map_location="cpu",
        ).half()
        t2 = torch.load(
            tensor_dir / f"run2_block_input_{args.layer}_{args.count}.pt",
            map_location="cpu",
        )
    elif args.block_output:
        t1 = torch.load(
            tensor_dir / f"run1_block_output_{args.layer}_{args.count}.pt",
            map_location="cpu",
        ).half()
        t2 = torch.load(
            tensor_dir / f"run2_block_output_{args.layer}_{args.count}.pt",
            map_location="cpu",
        )
    elif args.block:
        compare_layers(
            tensor_dir / f"run1_block_{args.layer}_{args.count}.pt",
            tensor_dir / f"run2_block_{args.layer}_{args.count}.pt",
        )
        exit()

    print(t1.dtype, t1.shape)
    print(t2.dtype, t2.shape)

    if t1.ndim == 3 and t2.ndim == 2:
        t2 = t2.unsqueeze(0)

    print(torch.equal(t1, t2))  # exact match
    print(torch.allclose(t1, t2, atol=1e-6))  # approximate match
    print((t1 - t2).abs().max())  # max absolute difference
    print((t1 - t2).abs().mean())  # mean absolute difference
