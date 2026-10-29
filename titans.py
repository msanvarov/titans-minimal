"""Titans: Learning to Memorize at Test Time, arXiv:2501.00663.

A neural long-term memory that is *trained by gradient descent during the
forward pass*. The memory is a small MLP; each incoming token gives it an
associative-recall task, and how badly it fails -- the "surprise" -- is the
gradient it takes a step along:

    loss_t    = || M_{t-1}(k_t) - v_t ||^2        k_t = x_t W_K,  v_t = x_t W_V
    S_t       = eta_t * S_{t-1} - theta_t * grad(loss_t)      momentary + past surprise
    M_t       = (1 - alpha_t) * M_{t-1} + S_t                 alpha_t forgets
    y_t       = M_t(q_t)                                      q_t = x_t W_Q

eta_t, theta_t and alpha_t are produced from the token itself, so the memory
decides per token how much to be surprised and how much to forget.

The outer parameters (W_K, W_V, W_Q, the gate heads and the memory's *initial*
weights) are learned normally -- the inner gradient is taken with
`create_graph=True`, so backprop flows through the test-time updates.

Three architectures from the paper: MemoryAsContext (MAC), MemoryAsGate (MAG)
and MemoryAsLayer (MAL).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sliding_window_attention(q, k, v, window):
    """Causal attention restricted to the last `window` tokens.

    Written out rather than calling `F.scaled_dot_product_attention`: the
    memory's inner update is differentiated with `create_graph=True`, and the
    fused CPU attention kernel has no double-backward.
    """
    T = q.shape[-2]
    t = torch.arange(T, device=q.device)
    mask = (t[:, None] >= t[None, :]) & (t[:, None] - t[None, :] < window)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return logits.masked_fill(~mask, float("-inf")).softmax(-1) @ v


class NeuralMemory(nn.Module):
    """The paper's long-term memory module (Section 3.1).

    Args:
        dim: model width.
        mem_dim: width the memory operates in (defaults to `dim`).
        depth: layers in the memory MLP. depth=1 is a linear memory, which
            recovers a DeltaNet-style linear-attention update; >=2 is what the
            paper argues you actually want.
        chunk_size: tokens per inner gradient step. 1 is the exact recurrence
            in the equations above; larger is the paper's chunk-wise
            parallelisation, and is what makes this runnable.
    """

    def __init__(self, dim, mem_dim=None, depth=2, chunk_size=16, hidden_mult=2):
        super().__init__()
        mem_dim = mem_dim or dim
        self.dim, self.mem_dim, self.chunk_size = dim, mem_dim, chunk_size

        self.to_k = nn.Linear(dim, mem_dim, bias=False)
        self.to_v = nn.Linear(dim, mem_dim, bias=False)
        self.to_q = nn.Linear(dim, mem_dim, bias=False)

        widths = [mem_dim] + [mem_dim * hidden_mult] * (depth - 1) + [mem_dim]
        self.w = nn.ParameterList()
        self.b = nn.ParameterList()
        for i in range(depth):
            self.w.append(nn.Parameter(torch.randn(widths[i], widths[i + 1])
                                       / math.sqrt(widths[i])))
            self.b.append(nn.Parameter(torch.zeros(widths[i + 1])))

        # data-dependent gates: alpha forgets, theta is the inner lr, eta is
        # the momentum on past surprise
        self.to_gates = nn.Linear(dim, 3)
        with torch.no_grad():
            # alpha ~ 0.018 (forget slowly), theta ~ 0.3, eta = 0.5.
            # theta is the one to turn down if the inner loop ever diverges --
            # it is a real gradient step on a freshly initialised MLP.
            self.to_gates.bias.copy_(torch.tensor([-4.0, -1.05, 0.0]))

    def _apply_memory(self, state, x):
        """Forward the memory MLP. `state` holds batched weights (B, in, out)."""
        w, b = state
        for i, (wi, bi) in enumerate(zip(w, b)):
            x = x @ wi + bi.unsqueeze(-2)
            if i < len(w) - 1:
                x = F.silu(x)
        return x

    def init_state(self, batch):
        """Memory at t=0: the learned initial weights, one copy per sequence."""
        w = [p.unsqueeze(0).repeat(batch, *([1] * p.ndim)) for p in self.w]
        b = [p.unsqueeze(0).repeat(batch, *([1] * p.ndim)) for p in self.b]
        return (w, b)

    def forward(self, x, state=None, return_state=False):
        """x: (B, T, dim) -> (B, T, mem_dim) retrieved values."""
        B, T, _ = x.shape
        state = self.init_state(B) if state is None else state
        surprise = ([torch.zeros_like(t) for t in state[0]],
                    [torch.zeros_like(t) for t in state[1]])

        k, v, q = self.to_k(x), self.to_v(x), self.to_q(x)
        gates = self.to_gates(x)
        alpha = gates[..., 0].sigmoid()
        theta = F.softplus(gates[..., 1])
        eta = gates[..., 2].sigmoid()

        out = []
        for s in range(0, T, self.chunk_size):
            e = min(s + self.chunk_size, T)
            # retrieve with the memory as it stands *before* this chunk: a token
            # must never be able to read a memory that already absorbed it
            out.append(self._apply_memory(state, q[:, s:e]))

            # ---- inner step: one gradient of the associative loss -----------
            flat = [t for t in state[0]] + [t for t in state[1]]
            # mean over features keeps the step size independent of mem_dim
            loss = (self._apply_memory(state, k[:, s:e]) - v[:, s:e]).pow(2).mean(-1).mean(-1)
            grads = torch.autograd.grad(loss.sum(), flat, create_graph=self.training)

            a = alpha[:, s:e].mean(-1)
            # the chunk's loss is a mean over its tokens, so one step has to
            # carry the weight of all of them -- without this, chunk_size
            # silently rescales the effective learning rate
            th = theta[:, s:e].mean(-1) * (e - s)
            et = eta[:, s:e].mean(-1)
            new_w, new_b, new_sw, new_sb = [], [], [], []
            for j, tensor in enumerate(flat):
                g = grads[j]
                prev = surprise[0][j] if j < len(state[0]) else surprise[1][j - len(state[0])]
                shape = (-1,) + (1,) * (tensor.ndim - 1)
                # S_t = eta_t S_{t-1} - theta_t grad
                s_new = et.view(shape) * prev - th.view(shape) * g
                # M_t = (1 - alpha_t) M_{t-1} + S_t
                t_new = (1 - a.view(shape)) * tensor + s_new
                if j < len(state[0]):
                    new_w.append(t_new)
                    new_sw.append(s_new)
                else:
                    new_b.append(t_new)
                    new_sb.append(s_new)
            state, surprise = (new_w, new_b), (new_sw, new_sb)

        y = torch.cat(out, dim=1)
        return (y, state) if return_state else y


class PersistentMemory(nn.Module):
    """Learnable, input-independent tokens prepended to the sequence (Sec. 3.2).

    They carry task knowledge and absorb the attention mass that would
    otherwise pile onto the first real token.
    """

    def __init__(self, dim, n_tokens=4):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(n_tokens, dim) * 0.02)

    def forward(self, x):
        return torch.cat([self.tokens.expand(x.shape[0], -1, -1), x], dim=1)


class _Attn(nn.Module):
    def __init__(self, dim, heads, window):
        super().__init__()
        self.h, self.dh, self.window = heads, dim // heads, window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        o = sliding_window_attention(q, k, v, self.window)
        return self.proj(o.transpose(1, 2).reshape(B, T, D))


class MemoryAsContext(nn.Module):
    """MAC: retrieve from memory, hand the result to attention as extra context.

    Per chunk: query the memory with the chunk, prepend [persistent, retrieved]
    to the chunk, attend over that, then update the memory with the attention
    output rather than the raw input.
    """

    def __init__(self, dim, heads=4, n_persist=4, chunk_size=16, **mem_kw):
        super().__init__()
        self.chunk_size = chunk_size
        self.persist = PersistentMemory(dim, n_persist)
        self.memory = NeuralMemory(dim, mem_dim=dim, chunk_size=chunk_size, **mem_kw)
        self.attn = _Attn(dim, heads, window=chunk_size * 4)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, _ = x.shape
        n_p = self.persist.tokens.shape[0]
        state, out = self.memory.init_state(B), []
        for s in range(0, T, self.chunk_size):
            chunk = x[:, s: s + self.chunk_size]
            # read the memory as of the previous chunk, use it as extra context
            retrieved = self.memory._apply_memory(state, self.memory.to_q(chunk))
            ctx = self.persist(torch.cat([retrieved, chunk], dim=1))
            attended = self.attn(self.norm(ctx))[:, n_p + chunk.shape[1]:]
            # the paper updates memory from the attention output, not the input
            _, state = self.memory(attended, state=state, return_state=True)
            # o_t = y_t (+) M*(y_t), with M* the memory that just absorbed y_t
            out.append(attended + self.memory._apply_memory(
                state, self.memory.to_q(attended)))
        return torch.cat(out, dim=1)


class MemoryAsGate(nn.Module):
    """MAG: sliding-window attention and memory as two branches, gated together."""

    def __init__(self, dim, heads=4, n_persist=4, window=32, **mem_kw):
        super().__init__()
        self.persist = PersistentMemory(dim, n_persist)
        self.attn = _Attn(dim, heads, window)
        self.memory = NeuralMemory(dim, mem_dim=dim, **mem_kw)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        n_p = self.persist.tokens.shape[0]
        h = self.norm(self.persist(x))
        short = self.attn(h)[:, n_p:]
        long = self.memory(x)
        g = self.gate(x).sigmoid()
        return g * short + (1 - g) * long


class MemoryAsLayer(nn.Module):
    """MAL: memory as an ordinary layer in front of sliding-window attention."""

    def __init__(self, dim, heads=4, n_persist=4, window=32, **mem_kw):
        super().__init__()
        self.persist = PersistentMemory(dim, n_persist)
        self.memory = NeuralMemory(dim, mem_dim=dim, **mem_kw)
        self.attn = _Attn(dim, heads, window)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        n_p = self.persist.tokens.shape[0]
        h = x + self.memory(x)
        return self.attn(self.norm(self.persist(h)))[:, n_p:]
