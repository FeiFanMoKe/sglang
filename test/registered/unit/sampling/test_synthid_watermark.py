"""Unit tests for SynthID-Text watermarking in srt/sampling/watermarking."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import math

import pytest
import torch

from sglang.srt.sampling.watermarking.config import (
    WatermarkConfigError,
    load_watermark_config,
)
from sglang.srt.sampling.watermarking.core import (
    _hash_contexts,
    _synthid_reweight_probs,
    _truncate_probabilities,
    _watermark_hash32_layer_torch,
    force_synthid_tokens,
    select_synthid_tokens_torch,
)
from sglang.srt.sampling.watermarking.detector import (
    _EXACT_BINOMIAL_MAX_TRIALS,
    _binomial_log_survival,
    _normal_log_survival,
    detect_synthid,
    hash_context,
    watermark_hash_layer,
)

KEY = "2a"
KEY_INT = int(KEY, 16)
DEPTH = 30
VOCAB_SIZE = 512
CONTEXT_WINDOW = 4


def _random_probs(batch, seed):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, VOCAB_SIZE, generator=generator)
    return _truncate_probabilities(
        logits,
        torch.ones(batch),
        torch.full((batch,), VOCAB_SIZE, dtype=torch.int32),
        torch.ones(batch),
        torch.zeros(batch),
    )


def test_layer_hash_torch_matches_scalar_detector():
    """Generation and detection must derive identical g-values."""
    contexts = torch.randint(0, VOCAB_SIZE, (3, CONTEXT_WINDOW))
    context_hashes = _hash_contexts(contexts, torch.full((3,), CONTEXT_WINDOW))
    keys = torch.tensor([KEY_INT, 0, -(2**63)], dtype=torch.int64)
    token_ids = torch.arange(VOCAB_SIZE)
    for layer in (0, 1, DEPTH - 1):
        hashed = _watermark_hash32_layer_torch(keys, context_hashes, token_ids, layer)
        for row in range(3):
            context = tuple(int(t) for t in contexts[row])
            from sglang.srt.sampling.watermarking.detector import hash_context

            context_hash = hash_context(context)
            expected = watermark_hash_layer(
                int(keys[row]), context_hash, 7, layer
            )
            assert hashed[row, 7].item() == expected


def test_g_values_balanced_across_layers():
    contexts = torch.randint(0, VOCAB_SIZE, (64, CONTEXT_WINDOW))
    context_hashes = _hash_contexts(contexts, torch.full((64,), CONTEXT_WINDOW))
    keys = torch.full((64,), KEY_INT, dtype=torch.int64)
    token_ids = torch.arange(VOCAB_SIZE)
    for layer in range(5):
        hashed = _watermark_hash32_layer_torch(keys, context_hashes, token_ids, layer)
        fraction_ones = (hashed >= (1 << 31)).double().mean().item()
        assert abs(fraction_ones - 0.5) < 0.02


def test_synthid_sampling_is_detectable():
    keys = torch.tensor([KEY_INT], dtype=torch.int64)
    token_ids: list[int] = []
    generator = torch.Generator().manual_seed(0)
    for _ in range(400):
        probs = _random_probs(1, seed=int(torch.randint(0, 2**31, (1,), generator=generator)))
        prefix = ([-1] * CONTEXT_WINDOW + token_ids)[-CONTEXT_WINDOW:]
        context_hashes = _hash_contexts(
            torch.tensor([prefix]), torch.tensor([CONTEXT_WINDOW])
        )
        token_ids.append(
            select_synthid_tokens_torch(probs, context_hashes, keys, DEPTH).item()
        )
    result = detect_synthid(token_ids, key=KEY, depth=DEPTH)
    assert result.p_value < 0.01
    assert result.z_score > 5


def test_unwatermarked_text_is_not_flagged():
    generator = torch.Generator().manual_seed(0)
    token_ids = torch.randint(0, VOCAB_SIZE, (400,), generator=generator).tolist()
    result = detect_synthid(token_ids, key=KEY, depth=DEPTH)
    assert result.p_value > 0.01


def test_wrong_key_does_not_detect():
    keys = torch.tensor([KEY_INT], dtype=torch.int64)
    token_ids: list[int] = []
    generator = torch.Generator().manual_seed(0)
    for _ in range(400):
        probs = _random_probs(1, seed=int(torch.randint(0, 2**31, (1,), generator=generator)))
        prefix = ([-1] * CONTEXT_WINDOW + token_ids)[-CONTEXT_WINDOW:]
        context_hashes = _hash_contexts(
            torch.tensor([prefix]), torch.tensor([CONTEXT_WINDOW])
        )
        token_ids.append(
            select_synthid_tokens_torch(probs, context_hashes, keys, DEPTH).item()
        )
    result = detect_synthid(token_ids, key="2b", depth=DEPTH)
    assert result.p_value > 0.01


def test_force_synthid_tokens_rewrites_only_eligible_rows():
    batch = 4
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(batch, VOCAB_SIZE, generator=generator)
    original = logits.clone()
    context_hashes = _hash_contexts(
        torch.randint(0, VOCAB_SIZE, (batch, CONTEXT_WINDOW)),
        torch.full((batch,), CONTEXT_WINDOW),
    )
    eligible = torch.tensor([True, False, True, False])
    keys = torch.full((batch,), KEY_INT, dtype=torch.int64)
    selected = force_synthid_tokens(
        logits,
        context_hashes,
        eligible,
        torch.ones(batch, 1),
        torch.full((batch,), VOCAB_SIZE, dtype=torch.int32),
        torch.ones(batch),
        torch.zeros(batch),
        keys,
        DEPTH,
    )
    assert selected.tolist() == [True, False, True, False]
    for row in range(batch):
        if eligible[row]:
            assert (logits[row] == 0.0).sum() == 1
            assert (logits[row] == -torch.inf).sum() == VOCAB_SIZE - 1
        else:
            assert torch.equal(logits[row], original[row])


def test_config_validation():
    config = load_watermark_config(
        '{"key": "2a", "algorithm": "synthid", "synthid_depth": 30}'
    )
    assert config.algorithm == "synthid"
    assert config.synthid_depth == 30
    with pytest.raises(WatermarkConfigError):
        load_watermark_config('{"key": "2a", "algorithm": "unknown"}')
    with pytest.raises(WatermarkConfigError):
        load_watermark_config('{"key": "2a", "synthid_depth": 0}')
    # Dual-key Gumbel knobs are meaningless (and silently ignored) under
    # SynthID, so they must be rejected instead.
    with pytest.raises(WatermarkConfigError):
        load_watermark_config('{"key": "2a", "algorithm": "synthid", "key_b": "2b"}')
    with pytest.raises(WatermarkConfigError):
        load_watermark_config(
            '{"key": "2a", "algorithm": "synthid", "mixing_probability": 0.5}'
        )
    with pytest.raises(WatermarkConfigError):
        load_watermark_config('{"key": "2a", "synthid_candidate_cap": 0}')
    config = load_watermark_config(
        '{"key": "2a", "algorithm": "synthid", "synthid_candidate_cap": 1024}'
    )
    assert config.synthid_candidate_cap == 1024


def test_reweight_deterministic_and_draw_stochastic():
    """The reweighted distribution is fixed per (key, context) and ignores
    global RNG state; the inverse-CDF draw samples it with fresh randomness,
    so the same seed must reproduce a draw and different seeds should vary."""
    probs = _random_probs(4, seed=11)
    context_hashes = _hash_contexts(
        torch.randint(0, VOCAB_SIZE, (4, CONTEXT_WINDOW)),
        torch.full((4,), CONTEXT_WINDOW),
    )
    keys = torch.full((4,), KEY_INT, dtype=torch.int64)
    expected = _synthid_reweight_probs(probs, context_hashes, keys, DEPTH)
    torch.randint(0, 2**31, (1000,))  # perturb the global RNG
    assert torch.equal(expected, _synthid_reweight_probs(probs, context_hashes, keys, DEPTH))
    torch.manual_seed(0)
    first = select_synthid_tokens_torch(probs, context_hashes, keys, DEPTH)
    torch.manual_seed(0)
    assert torch.equal(first, select_synthid_tokens_torch(probs, context_hashes, keys, DEPTH))


def test_sampling_marginal_matches_base_distribution():
    """Over the key ensemble the sampled token follows the original
    distribution: reweighting tilts per key, sampling stays non-distortionary."""
    base = torch.tensor([0.5, 0.3, 0.15, 0.05])
    probs = base.unsqueeze(0)
    counts = torch.zeros(4)
    trials = 4000
    generator = torch.Generator().manual_seed(17)
    torch.manual_seed(19)
    for _ in range(trials):
        key = int(torch.randint(0, 2**31, (1,), generator=generator))
        context_hash = torch.randint(0, 2**32, (1,), generator=generator)
        selected = select_synthid_tokens_torch(
            probs, context_hash, torch.tensor([key]), DEPTH
        )
        counts[selected.item()] += 1
    assert torch.allclose(counts / trials, base, atol=0.03)


def test_candidate_cap_exact_when_support_fits_cap():
    """Support <= cap must reproduce the uncapped update."""
    probs = _random_probs(2, seed=12)
    probs[:, VOCAB_SIZE // 2 :] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    context_hashes = _hash_contexts(
        torch.randint(0, VOCAB_SIZE, (2, CONTEXT_WINDOW)),
        torch.full((2,), CONTEXT_WINDOW),
    )
    keys = torch.full((2,), KEY_INT, dtype=torch.int64)
    exact = _synthid_reweight_probs(probs, context_hashes, keys, DEPTH)
    capped = _synthid_reweight_probs(
        probs, context_hashes, keys, DEPTH, candidate_cap=300
    )
    assert torch.allclose(capped, exact, atol=1e-6)


def test_candidate_cap_preserves_tail_shape_and_mass():
    """With support > cap the tail is scaled uniformly (neutral treatment)
    and the reweighted distribution stays normalized."""
    probs = _random_probs(2, seed=13)
    context_hashes = _hash_contexts(
        torch.randint(0, VOCAB_SIZE, (2, CONTEXT_WINDOW)),
        torch.full((2,), CONTEXT_WINDOW),
    )
    keys = torch.full((2,), KEY_INT, dtype=torch.int64)
    cap = 64
    out = _synthid_reweight_probs(
        probs, context_hashes, keys, DEPTH, candidate_cap=cap
    )
    assert torch.allclose(out.sum(dim=-1), torch.ones(2), atol=1e-5)
    top_indices = probs.topk(cap, dim=-1).indices
    tail_mask = torch.ones_like(probs, dtype=torch.bool).scatter_(
        -1, top_indices, False
    )
    ratios = out[tail_mask] / probs[tail_mask]
    for row_ratios in ratios.split(VOCAB_SIZE - cap):
        assert row_ratios.max().item() / row_ratios.min().item() < 1.001


def test_binomial_normal_approximation_continuous_at_cutoff():
    """Above _EXACT_BINOMIAL_MAX_TRIALS the detector switches to the normal
    approximation; both must agree at the boundary."""
    trials = _EXACT_BINOMIAL_MAX_TRIALS
    count = trials // 2 + 500
    exact = _binomial_log_survival(count, trials)
    approx = _normal_log_survival(
        (count - 0.5 - trials / 2) / math.sqrt(trials / 4)
    )
    assert math.exp(approx) == pytest.approx(math.exp(exact), rel=0.01)


def test_config_synthid_compile_flag():
    config = load_watermark_config(
        '{"key": "2a", "algorithm": "synthid", "synthid_compile": true}'
    )
    assert config.synthid_compile is True
    with pytest.raises(WatermarkConfigError):
        load_watermark_config('{"key": "2a", "synthid_compile": "yes"}')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compile path uses CUDA")
def test_compiled_reweight_allclose_and_deterministic():
    """The compiled reweight may reorder fp32 reductions (not bit-exact),
    but must stay close to eager and deterministic per (key, context)."""
    from sglang.srt.sampling.watermarking.core import (
        _get_compiled_synthid_reweight,
        _synthid_reweight_probs,
    )

    probs = _random_probs(4, seed=21).cuda()
    context_hashes = _hash_contexts(
        torch.randint(0, VOCAB_SIZE, (4, CONTEXT_WINDOW)),
        torch.full((4,), CONTEXT_WINDOW),
    ).cuda()
    keys = torch.full((4,), KEY_INT, dtype=torch.int64, device="cuda")
    compiled = _get_compiled_synthid_reweight()
    expected = _synthid_reweight_probs(probs, context_hashes, keys, DEPTH, 256)
    actual = compiled(probs, context_hashes, keys, DEPTH, 256)
    assert torch.allclose(expected, actual, rtol=1e-5, atol=1e-7)
    rerun = compiled(probs, context_hashes, keys, DEPTH, 256)
    assert torch.equal(actual, rerun)
