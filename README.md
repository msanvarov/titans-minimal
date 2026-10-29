# titans-minimal

A minimal PyTorch implementation of **Titans** — a neural memory that trains
itself by gradient descent *during the forward pass*.

> **Titans: Learning to Memorize at Test Time**
> Behrouz, Zhong, Mirrokni (Google) — [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)

The memory is a small MLP. Every token hands it an associative-recall task, and
how badly it fails — the **surprise** — is the gradient it steps along:

```
loss_t = ‖ M_{t-1}(k_t) − v_t ‖²              k_t = x_t W_K,  v_t = x_t W_V
S_t    = η_t · S_{t-1} − θ_t · ∇loss_t         past surprise + momentary surprise
M_t    = (1 − α_t) · M_{t-1} + S_t             α_t forgets
y_t    = M_t(q_t)                              q_t = x_t W_Q
```

`η_t`, `θ_t` and `α_t` are produced from the token itself, so the memory decides
per token how surprised to be and how much to forget. The inner gradient is
taken with `create_graph=True`, so the outer parameters — `W_K`, `W_V`, `W_Q`,
the gate heads, and the memory's *initial* weights — learn by backprop through
the test-time updates.

## Install

```bash
pip install torch
```

## Use

```python
import torch
from titans import NeuralMemory, MemoryAsContext, MemoryAsGate, MemoryAsLayer

mem = NeuralMemory(dim=512, depth=2, chunk_size=16)
y = mem(torch.randn(2, 1024, 512))           # retrieved values, (2, 1024, 512)
y, state = mem(x, return_state=True)         # keep the memory for the next call
y2 = mem(next_chunk, state=state)            # ...and continue from it
```

All three architectures from the paper:

| class | shape of the idea |
|---|---|
| `MemoryAsContext` | query the memory per chunk, prepend the result to the chunk as extra context for attention, then update the memory from the attention output |
| `MemoryAsGate` | sliding-window attention and memory as two parallel branches, mixed by a learned gate |
| `MemoryAsLayer` | memory as an ordinary layer in front of sliding-window attention |

`PersistentMemory` adds the learnable input-independent tokens from Section 3.2,
which carry task knowledge and soak up attention mass that would otherwise pile
onto the first real token.

### Two things that will bite you

**Inference needs grad mode.** The inner update is a real `autograd.grad` call,
so `torch.no_grad()` breaks it. Wrap inference in `torch.enable_grad()`.

**`chunk_size` rescales the step.** A chunk's loss is a *mean* over its tokens,
so the implementation multiplies `θ_t` by the chunk length — without that,
changing `chunk_size` silently changes the effective learning rate. `θ` itself
is the knob to turn down if the inner loop ever diverges; it is a genuine
gradient step on a freshly initialised MLP, and large values do blow up.

## Verify

```bash
python test_titans.py   # 12 checks, ~20s
python train.py         # memory vs attention on long-range recall, ~10s
```

The headline check runs an **untrained** memory over a random sequence and
measures associative recall before and after. Nothing is optimized; the only
weight updates are the ones the memory makes on itself while reading:

```
recall MSE 0.4457 -> 0.1731
```

The rest pin down behaviour that is easy to get silently wrong:

- **causality** — retrieval uses the memory as of the previous chunk, so
  perturbing later tokens leaves earlier outputs identical. (All three
  architectures are checked too; getting MAC wrong here is easy.)
- **the outer gradient really flows** through the inner update — every
  parameter, including the memory's initial weights, receives gradient
- **gates do what they say** — `α→1` discards the previous memory entirely,
  `θ→0` freezes the state bit-for-bit
- **`α > 0` produces recency bias** — the newest quarter of a 256-token sequence
  recalls better than the oldest, across seeds
- `depth=1` (a linear memory, i.e. a DeltaNet-style update) still memorizes

`train.py` hides a needle outside an 8-token attention window. MAG reaches
**1.000** accuracy; the identical model with the memory removed stays at
**0.117** against a chance baseline of 0.125 — the memory is the only path to
the needle, and it finds it.

## Two findings worth recording

**`chunk_size=1` memorizes *worse* than chunked updates** (~0.65x the initial
error vs ~0.39x at `chunk_size=4`), even though it is the literal per-token
recurrence in the paper. Single-token gradients partly undo each other; a
chunk's mean gradient is a cleaner direction. Worth knowing before reaching for
`chunk_size=1` in the name of fidelity.

**The memory saturates.** Recall error at T=16, 64 and 256 all land around
0.18–0.19. Reading more tokens stops helping, because the MLP is a fixed size
and `α_t` keeps forgetting. Capacity, not sequence length, is the binding
constraint.

## Scope

The inner loop runs chunk-by-chunk in Python, so this is readable and slow —
no Section 3.2 parallel scan, no `torch.func.vmap` over the memory, no custom
kernel. Fine for research at small scale; not a drop-in for a long-context
production model.

Also not implemented: the paper's full Titans blocks (convolution before
projections, normalization choices, RoPE), and the deeper memory variants from
the ablations.

## License

MIT
