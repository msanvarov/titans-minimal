"""Self-checks for Titans. Run: python test_titans.py"""

import torch

from titans import (MemoryAsContext, MemoryAsGate, MemoryAsLayer, NeuralMemory,
                    PersistentMemory)


def recall_error(mem, state, x):
    """Mean squared error of M(k_i) vs v_i over the tokens the memory has seen."""
    k, v = mem.to_k(x), mem.to_v(x)
    return (mem._apply_memory(state, k) - v).pow(2).mean().item()


def test_it_memorizes_at_test_time():
    """The whole thesis: recall improves during the forward pass, with no training.

    Nothing here is optimized. The only weight updates are the ones the memory
    performs on itself while reading the sequence.
    """
    torch.manual_seed(0)
    mem = NeuralMemory(32, chunk_size=4).eval()
    x = torch.randn(4, 128, 32)
    before = recall_error(mem, mem.init_state(4), x)
    _, after = mem(x, return_state=True)
    err = recall_error(mem, after, x)
    assert err < 0.5 * before, f"no memorization: {before:.4f} -> {err:.4f}"
    print(f"    recall MSE {before:.4f} -> {err:.4f}")


def test_exact_recurrence_also_memorizes():
    """chunk_size=1 is the literal per-token recurrence in the equations.

    It memorizes, but *less well* than chunked updates (~0.65x the initial
    error vs ~0.39x at chunk_size=4): single-token gradients partly undo each
    other, where a chunk's mean gradient is a cleaner direction. Worth knowing
    before you reach for chunk_size=1 in the name of fidelity.
    """
    torch.manual_seed(0)
    mem = NeuralMemory(16, chunk_size=1).eval()
    x = torch.randn(2, 64, 16)
    before = recall_error(mem, mem.init_state(2), x)
    _, exact = mem(x, return_state=True)
    exact_err = recall_error(mem, exact, x)
    assert exact_err < 0.8 * before, (before, exact_err)

    torch.manual_seed(0)
    chunked = NeuralMemory(16, chunk_size=4).eval()
    _, st = chunked(x, return_state=True)
    assert recall_error(chunked, st, x) < exact_err


def test_memory_saturates_rather_than_improving_forever():
    """Fixed-capacity memory: recall drops fast, then plateaus.

    A finding worth recording, not a flaw -- error at T=16, 64 and 256 all sit
    around 0.18-0.19 against an initial 0.42-0.45. Reading more tokens does not
    keep helping, because the MLP has a fixed size and alpha_t keeps forgetting.
    """
    for seed in range(3):
        torch.manual_seed(seed)
        mem = NeuralMemory(32, chunk_size=4).eval()
        x = torch.randn(4, 256, 32)
        init = recall_error(mem, mem.init_state(4), x)
        errs = []
        for T in (16, 64, 256):
            _, st = mem(x[:, :T], return_state=True)
            errs.append(recall_error(mem, st, x[:, :T]))
        assert all(e < 0.6 * init for e in errs), (seed, init, errs)
        assert max(errs) / min(errs) < 1.3, (seed, errs)  # plateau, not a trend


def test_forget_gate_produces_recency_bias():
    """alpha_t > 0 means old associations decay: the newest tokens recall best."""
    for seed in range(3):
        torch.manual_seed(seed)
        mem = NeuralMemory(32, chunk_size=4).eval()
        x = torch.randn(4, 256, 32)
        _, st = mem(x, return_state=True)
        oldest = recall_error(mem, st, x[:, :64])
        newest = recall_error(mem, st, x[:, 192:])
        assert newest < oldest, (seed, oldest, newest)


def test_retrieval_is_causal():
    """A token must not read a memory that has already absorbed it."""
    torch.manual_seed(0)
    mem = NeuralMemory(32, chunk_size=4).eval()
    x = torch.randn(1, 32, 32)
    y = mem(x)
    x2 = x.clone()
    x2[:, 16:] += torch.randn_like(x2[:, 16:]) * 5
    y2 = mem(x2)
    assert torch.allclose(y[:, :16], y2[:, :16], atol=1e-5), \
        (y[:, :16] - y2[:, :16]).abs().max()


def test_outer_gradient_flows_through_the_inner_update():
    """The initial memory weights can only get gradient via create_graph=True."""
    torch.manual_seed(0)
    mem = NeuralMemory(32, chunk_size=4).train()
    mem(torch.randn(2, 32, 32)).pow(2).mean().backward()
    for name, p in mem.named_parameters():
        assert p.grad is not None, name
        assert p.grad.abs().sum() > 0, name


def test_full_forget_gate_discards_the_memory():
    """alpha_t -> 1 must leave M_t = S_t, i.e. nothing of M_{t-1} survives."""
    torch.manual_seed(0)
    mem = NeuralMemory(16, chunk_size=16).eval()
    with torch.no_grad():
        mem.to_gates.weight.zero_()
        mem.to_gates.bias.copy_(torch.tensor([30.0, -2.0, 0.0]))  # alpha ~ 1
    x = torch.randn(1, 16, 16)
    _, state = mem(x, return_state=True)
    # with M_{t-1} fully forgotten, the surviving state is -theta * grad, whose
    # bias terms are tiny -- certainly not the learned initial bias
    init = mem.init_state(1)
    assert not torch.allclose(state[0][0], init[0][0], atol=1e-3)
    for i, w in enumerate(state[0]):
        assert w.abs().max() < init[0][i].abs().max(), i


def test_zero_lr_freezes_the_memory():
    """theta_t -> 0 means no inner step at all: the state must not move."""
    torch.manual_seed(0)
    mem = NeuralMemory(16, chunk_size=8).eval()
    with torch.no_grad():
        mem.to_gates.weight.zero_()
        mem.to_gates.bias.copy_(torch.tensor([-30.0, -30.0, 0.0]))  # alpha~0, theta~0
    x = torch.randn(1, 32, 16)
    _, state = mem(x, return_state=True)
    init = mem.init_state(1)
    for a, b in zip(state[0] + state[1], init[0] + init[1]):
        assert torch.allclose(a, b, atol=1e-8), (a - b).abs().max()


def test_linear_memory_depth_one():
    """depth=1 is a linear memory -- a DeltaNet-style update. Still memorizes."""
    torch.manual_seed(0)
    mem = NeuralMemory(16, depth=1, chunk_size=4).eval()
    x = torch.randn(2, 64, 16)
    before = recall_error(mem, mem.init_state(2), x)
    _, after = mem(x, return_state=True)
    assert recall_error(mem, after, x) < before


def test_persistent_memory_prepends_learnable_tokens():
    pm = PersistentMemory(8, n_tokens=3)
    out = pm(torch.randn(2, 5, 8))
    assert out.shape == (2, 8, 8)
    assert torch.allclose(out[0, :3], pm.tokens)
    out.sum().backward()
    assert pm.tokens.grad.abs().sum() > 0


def test_all_three_variants_train_and_stay_causal():
    torch.manual_seed(0)
    for cls in (MemoryAsContext, MemoryAsGate, MemoryAsLayer):
        mod = cls(32, chunk_size=8) if cls is MemoryAsContext else cls(32)
        mod.train()
        x = torch.randn(2, 32, 32)
        y = mod(x)
        assert y.shape == x.shape, (cls.__name__, y.shape)
        y.pow(2).mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in mod.parameters()), cls.__name__

        mod.eval()
        x2 = x.clone()
        x2[:, 24:] += 5 * torch.randn_like(x2[:, 24:])
        a, b = mod(x), mod(x2)
        assert torch.allclose(a[:, :16], b[:, :16], atol=1e-4), \
            (cls.__name__, (a[:, :16] - b[:, :16]).abs().max())


def test_inference_needs_grad_mode():
    """Documented gotcha: the inner step is a real gradient, so no_grad breaks it."""
    mem = NeuralMemory(16, chunk_size=8).eval()
    try:
        with torch.no_grad():
            mem(torch.randn(1, 16, 16))
    except RuntimeError:
        pass  # expected -- use torch.enable_grad() or inference_mode=False
    else:
        raise AssertionError("expected no_grad to break the inner update")
    with torch.enable_grad():
        assert mem(torch.randn(1, 16, 16)).shape == (1, 16, 16)


if __name__ == "__main__":
    for fn in [
        test_it_memorizes_at_test_time,
        test_exact_recurrence_also_memorizes,
        test_memory_saturates_rather_than_improving_forever,
        test_forget_gate_produces_recency_bias,
        test_retrieval_is_causal,
        test_outer_gradient_flows_through_the_inner_update,
        test_full_forget_gate_discards_the_memory,
        test_zero_lr_freezes_the_memory,
        test_linear_memory_depth_one,
        test_persistent_memory_prepends_learnable_tokens,
        test_all_three_variants_train_and_stay_causal,
        test_inference_needs_grad_mode,
    ]:
        fn()
        print(f"ok  {fn.__name__}")
    print("\nall checks passed")
