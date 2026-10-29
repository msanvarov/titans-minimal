"""Does the test-time memory actually buy long-range recall? CPU, ~1-2 min.

Task ("needle recall"): filler tokens with exactly one needle from a disjoint
range, hidden *outside* the attention window. The model must emit it at the
last position.

Two models, same width and depth:
  * `window` -- sliding-window attention only. The needle is out of range, so
    this is the control and it should stay at chance.
  * `mag`    -- the same attention plus Titans' neural memory, gated (MAG).
    The memory is the only path to the needle.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from titans import MemoryAsGate, PersistentMemory, _Attn

T, WINDOW = 64, 8
FILLER = (2, 10)
NEEDLE = (10, 18)
VOCAB = NEEDLE[1]
N_CLASSES = NEEDLE[1] - NEEDLE[0]


def batch(n):
    x = torch.randint(*FILLER, (n, T))
    pos = torch.randint(0, T - WINDOW - 1, (n,))
    needle = torch.randint(*NEEDLE, (n,))
    x[torch.arange(n), pos] = needle
    x[:, -1] = 0
    return x, needle - NEEDLE[0]


class WindowOnly(nn.Module):
    """Control: no memory. The needle is outside the window by construction."""

    def __init__(self, dim):
        super().__init__()
        self.persist = PersistentMemory(dim, 4)
        self.attn = _Attn(dim, heads=4, window=WINDOW)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h):
        return self.attn(self.norm(self.persist(h)))[:, 4:]


class Model(nn.Module):
    def __init__(self, kind, dim=48):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, dim)
        self.pos = nn.Parameter(torch.randn(T, dim) * 0.02)
        self.mixer = (
            MemoryAsGate(dim, heads=4, window=WINDOW, chunk_size=8, depth=2)
            if kind == "mag" else WindowOnly(dim)
        )
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.head = nn.Linear(dim, N_CLASSES)

    def forward(self, x):
        h = self.embed(x) + self.pos[: x.shape[1]]
        h = h + self.mixer(self.n1(h))
        h = h + self.mlp(self.n2(h))
        return self.head(h[:, -1])


def run(kind, steps, bs, lr, seed):
    torch.manual_seed(seed)
    model = Model(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    acc = 0.0
    for step in range(1, steps + 1):
        x, y = batch(bs)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps:
            with torch.no_grad(), torch.enable_grad():
                xe, ye = batch(256)
                acc = (model(xe).argmax(-1) == ye).float().mean().item()
            print(f"  {kind:>7}  step {step:4d}  loss {loss.item():.3f}  acc {acc:.3f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"needle hidden outside a {WINDOW}-token attention window, T={T}\n")
    mag = run("mag", a.steps, a.bs, a.lr, a.seed)
    print()
    ctrl = run("window", a.steps, a.bs, a.lr, a.seed)

    chance = 1 / N_CLASSES
    print(f"\nchance {chance:.3f}   mag {mag:.3f}   window-only {ctrl:.3f}")
    assert mag > 3 * chance, f"memory failed to retrieve the needle ({mag:.3f})"
    assert mag > ctrl + 0.1, f"memory gave no advantage ({mag:.3f} vs {ctrl:.3f})"
    print("ok  the test-time memory reached what attention could not see")


if __name__ == "__main__":
    main()
