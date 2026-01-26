"""
Tests for TRM Loss Functions

Run with: pytest tests/test_losses.py -v
"""

import pytest
import torch
import torch.nn as nn
import math

from ..src.losses import (
    TokenCrossEntropyLoss,
    DiffWeightedLoss,
    IntermediateSupervisionLoss,
    CombinedCodeRepairLoss,
    compute_perplexity,
    compute_accuracy,
    compute_diff_accuracy,
    compute_edit_distance_metrics,
    compute_sequence_accuracy,
)


class TestTokenCrossEntropyLoss:
    """Tests for TokenCrossEntropyLoss."""

    @pytest.fixture
    def loss_fn(self):
        return TokenCrossEntropyLoss(label_smoothing=0.0, ignore_index=-100)

    def test_basic_forward(self, loss_fn):
        batch_size, height, width, vocab_size = 2, 8, 8, 100
        logits = torch.randn(batch_size, height, width, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, height, width))

        loss = loss_fn(logits, targets)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_label_smoothing(self):
        loss_fn_no_smooth = TokenCrossEntropyLoss(label_smoothing=0.0)
        loss_fn_smooth = TokenCrossEntropyLoss(label_smoothing=0.1)

        logits = torch.randn(2, 8, 8, 100)
        targets = torch.randint(0, 100, (2, 8, 8))

        loss_no_smooth = loss_fn_no_smooth(logits, targets)
        loss_smooth = loss_fn_smooth(logits, targets)

        # Label smoothing generally increases loss (distributes probability)
        # The losses should be different
        assert not torch.isclose(loss_no_smooth, loss_smooth)

    def test_ignore_index(self, loss_fn):
        logits = torch.randn(2, 8, 8, 100)
        targets = torch.randint(0, 100, (2, 8, 8))

        # Set some targets to ignore index
        targets[0, :4, :] = -100

        loss = loss_fn(logits, targets)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_reduction_none(self):
        loss_fn = TokenCrossEntropyLoss(reduction="none")
        logits = torch.randn(2, 8, 8, 100)
        targets = torch.randint(0, 100, (2, 8, 8))

        loss = loss_fn(logits, targets)

        # Should return per-token losses
        assert loss.shape == (2 * 8 * 8,)

    def test_reduction_sum(self):
        loss_fn_mean = TokenCrossEntropyLoss(reduction="mean")
        loss_fn_sum = TokenCrossEntropyLoss(reduction="sum")

        logits = torch.randn(2, 8, 8, 100)
        targets = torch.randint(0, 100, (2, 8, 8))

        loss_mean = loss_fn_mean(logits, targets)
        loss_sum = loss_fn_sum(logits, targets)

        # Sum should be mean * num_tokens
        n_tokens = 2 * 8 * 8
        assert torch.isclose(loss_sum, loss_mean * n_tokens, rtol=1e-4)

    def test_invalid_reduction(self):
        with pytest.raises(ValueError):
            TokenCrossEntropyLoss(reduction="invalid")

    def test_gradient_flow(self, loss_fn):
        logits = torch.randn(2, 8, 8, 100, requires_grad=True)
        targets = torch.randint(0, 100, (2, 8, 8))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestDiffWeightedLoss:
    """Tests for DiffWeightedLoss."""

    @pytest.fixture
    def loss_fn(self):
        return DiffWeightedLoss(diff_weight=5.0, base_weight=1.0)

    def test_compute_diff_mask(self, loss_fn):
        buggy = torch.tensor([[1, 2, 3], [4, 5, 6]])
        fixed = torch.tensor([[1, 2, 4], [4, 5, 6]])

        mask = loss_fn.compute_diff_mask(buggy, fixed)

        expected = torch.tensor([[False, False, True], [False, False, False]])
        assert torch.equal(mask, expected)

    def test_forward_without_diff_mask(self, loss_fn):
        logits = torch.randn(2, 8, 8, 100)
        targets = torch.randint(0, 100, (2, 8, 8))

        loss = loss_fn(logits, targets, diff_mask=None)

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_forward_with_diff_mask(self, loss_fn):
        logits = torch.randn(2, 8, 8, 100)
        targets = torch.randint(0, 100, (2, 8, 8))
        diff_mask = torch.zeros(2, 8, 8, dtype=torch.bool)
        diff_mask[:, :2, :] = True  # First 2 rows changed

        loss = loss_fn(logits, targets, diff_mask)

        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_diff_weighting_effect(self):
        # Create controlled scenario
        vocab_size = 10
        logits = torch.randn(1, 4, 4, vocab_size)
        targets = torch.randint(0, vocab_size, (1, 4, 4))

        # All changed vs none changed
        all_changed = torch.ones(1, 4, 4, dtype=torch.bool)
        none_changed = torch.zeros(1, 4, 4, dtype=torch.bool)

        loss_fn = DiffWeightedLoss(diff_weight=10.0, base_weight=1.0)

        loss_all_changed = loss_fn(logits, targets, all_changed)
        loss_none_changed = loss_fn(logits, targets, none_changed)

        # Both should be valid
        assert loss_all_changed.dim() == 0
        assert loss_none_changed.dim() == 0

    def test_ignore_index(self, loss_fn):
        logits = torch.randn(2, 8, 8, 100)
        targets = torch.randint(0, 100, (2, 8, 8))
        targets[0, :2, :] = -100  # Ignore first 2 rows

        diff_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        loss = loss_fn(logits, targets, diff_mask)

        assert not torch.isnan(loss)

    def test_gradient_flow(self, loss_fn):
        logits = torch.randn(2, 8, 8, 100, requires_grad=True)
        targets = torch.randint(0, 100, (2, 8, 8))
        diff_mask = torch.randint(0, 2, (2, 8, 8)).bool()

        loss = loss_fn(logits, targets, diff_mask)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestIntermediateSupervisionLoss:
    """Tests for IntermediateSupervisionLoss."""

    @pytest.fixture
    def loss_fn(self):
        return IntermediateSupervisionLoss(n_iterations=4)

    def test_uniform_weights(self):
        loss_fn = IntermediateSupervisionLoss(n_iterations=4, iteration_weights=None)
        weights = loss_fn.get_weights()

        assert len(weights) == 4
        assert torch.isclose(weights.sum(), torch.tensor(1.0))
        # Should be uniform: 0.25 each
        assert torch.allclose(weights, torch.tensor([0.25, 0.25, 0.25, 0.25]))

    def test_linear_weights(self):
        loss_fn = IntermediateSupervisionLoss(n_iterations=4, iteration_weights="linear")
        weights = loss_fn.get_weights()

        # Linear: 1, 2, 3, 4 normalized
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected = expected / expected.sum()

        assert torch.allclose(weights, expected)

    def test_exponential_weights(self):
        loss_fn = IntermediateSupervisionLoss(n_iterations=4, iteration_weights="exponential")
        weights = loss_fn.get_weights()

        # Exponential: 1, 2, 4, 8 normalized
        expected = torch.tensor([1.0, 2.0, 4.0, 8.0])
        expected = expected / expected.sum()

        assert torch.allclose(weights, expected)

    def test_custom_weights(self):
        custom = [0.1, 0.2, 0.3, 0.4]
        loss_fn = IntermediateSupervisionLoss(n_iterations=4, iteration_weights=custom)
        weights = loss_fn.get_weights()

        assert torch.isclose(weights.sum(), torch.tensor(1.0))

    def test_forward(self, loss_fn):
        vocab_size = 100
        all_logits = [torch.randn(2, 8, 8, vocab_size) for _ in range(4)]
        targets = torch.randint(0, vocab_size, (2, 8, 8))

        loss, info = loss_fn(all_logits, targets)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert "iteration_losses" in info
        assert "iteration_weights" in info
        assert len(info["iteration_losses"]) == 4

    def test_fewer_iterations(self, loss_fn):
        vocab_size = 100
        # Only 2 iterations instead of expected 4
        all_logits = [torch.randn(2, 8, 8, vocab_size) for _ in range(2)]
        targets = torch.randint(0, vocab_size, (2, 8, 8))

        loss, info = loss_fn(all_logits, targets)

        assert info["n_iterations"] == 2
        assert len(info["iteration_losses"]) == 2

    def test_learnable_weights(self):
        loss_fn = IntermediateSupervisionLoss(n_iterations=4, learnable_weights=True)

        # Check weights are parameters
        assert hasattr(loss_fn, "log_weights")
        assert loss_fn.log_weights.requires_grad

        vocab_size = 100
        all_logits = [torch.randn(2, 8, 8, vocab_size) for _ in range(4)]
        targets = torch.randint(0, vocab_size, (2, 8, 8))

        loss, _ = loss_fn(all_logits, targets)
        loss.backward()

        assert loss_fn.log_weights.grad is not None

    def test_gradient_flow(self, loss_fn):
        vocab_size = 100
        all_logits = [torch.randn(2, 8, 8, vocab_size, requires_grad=True) for _ in range(4)]
        targets = torch.randint(0, vocab_size, (2, 8, 8))

        loss, _ = loss_fn(all_logits, targets)
        loss.backward()

        for logits in all_logits:
            assert logits.grad is not None
            assert not torch.isnan(logits.grad).any()

    def test_invalid_weight_scheme(self):
        with pytest.raises(ValueError):
            IntermediateSupervisionLoss(n_iterations=4, iteration_weights="invalid")


class TestCombinedCodeRepairLoss:
    """Tests for CombinedCodeRepairLoss."""

    @pytest.fixture
    def loss_fn(self):
        return CombinedCodeRepairLoss(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            diff_weight=5.0,
            n_iterations=4,
        )

    def test_ce_loss_only(self):
        loss_fn = CombinedCodeRepairLoss(alpha=1.0, beta=0.0, gamma=0.0)

        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 8, 8))

        result = loss_fn(logits, targets)

        assert "total" in result
        assert "ce_loss" in result
        assert "diff_loss" not in result
        assert "intermediate_loss" not in result

    def test_with_diff_loss(self, loss_fn):
        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 8, 8))
        buggy_ids = torch.randint(0, vocab_size, (2, 8, 8))

        result = loss_fn(logits, targets, buggy_ids=buggy_ids)

        assert "diff_loss" in result
        assert "diff_mask_sum" in result

    def test_with_intermediate_loss(self, loss_fn):
        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 8, 8))
        all_logits = [torch.randn(2, 8, 8, vocab_size) for _ in range(4)]

        result = loss_fn(logits, targets, all_iteration_logits=all_logits)

        assert "intermediate_loss" in result
        assert "iteration_losses" in result
        assert "iteration_weights" in result

    def test_full_combined_loss(self, loss_fn):
        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 8, 8))
        buggy_ids = torch.randint(0, vocab_size, (2, 8, 8))
        all_logits = [torch.randn(2, 8, 8, vocab_size) for _ in range(4)]

        result = loss_fn(
            logits,
            targets,
            buggy_ids=buggy_ids,
            all_iteration_logits=all_logits,
        )

        assert "total" in result
        assert "ce_loss" in result
        assert "diff_loss" in result
        assert "intermediate_loss" in result

        # Verify total is weighted sum
        expected_total = (
            0.5 * result["ce_loss"]
            + 0.3 * result["diff_loss"]
            + 0.2 * result["intermediate_loss"]
        )
        assert torch.isclose(result["total"], expected_total, rtol=1e-4)

    def test_gradient_scales(self):
        loss_fn = CombinedCodeRepairLoss(
            alpha=1.0,
            beta=1.0,
            gamma=0.0,
            gradient_scales={"ce_loss": 0.5, "diff_loss": 2.0},
        )

        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (2, 8, 8))
        buggy_ids = torch.randint(0, vocab_size, (2, 8, 8))

        result = loss_fn(logits, targets, buggy_ids=buggy_ids)
        result["total"].backward()

        # Just verify gradients exist
        assert logits.grad is not None

    def test_gradient_flow_all_components(self, loss_fn):
        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (2, 8, 8))
        buggy_ids = torch.randint(0, vocab_size, (2, 8, 8))
        all_logits = [torch.randn(2, 8, 8, vocab_size, requires_grad=True) for _ in range(4)]

        result = loss_fn(
            logits,
            targets,
            buggy_ids=buggy_ids,
            all_iteration_logits=all_logits,
        )

        result["total"].backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

        for iter_logits in all_logits:
            assert iter_logits.grad is not None


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_perplexity(self):
        loss = torch.tensor(2.0)
        ppl = compute_perplexity(loss)

        expected = math.exp(2.0)
        assert torch.isclose(ppl, torch.tensor(expected))

    def test_compute_perplexity_zero_loss(self):
        loss = torch.tensor(0.0)
        ppl = compute_perplexity(loss)

        assert torch.isclose(ppl, torch.tensor(1.0))

    def test_compute_accuracy_perfect(self):
        vocab_size = 100
        batch_size = 2

        # Create logits where argmax equals targets
        targets = torch.randint(0, vocab_size, (batch_size, 8, 8))
        logits = torch.zeros(batch_size, 8, 8, vocab_size)

        for b in range(batch_size):
            for i in range(8):
                for j in range(8):
                    logits[b, i, j, targets[b, i, j]] = 10.0

        acc = compute_accuracy(logits, targets)
        assert torch.isclose(acc, torch.tensor(1.0))

    def test_compute_accuracy_random(self):
        logits = torch.randn(2, 8, 8, 100)
        targets = torch.randint(0, 100, (2, 8, 8))

        acc = compute_accuracy(logits, targets)

        # Random should give ~1% accuracy for vocab 100
        assert 0.0 <= acc.item() <= 1.0

    def test_compute_accuracy_with_ignore(self):
        vocab_size = 10
        logits = torch.zeros(1, 4, 4, vocab_size)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)

        # Make predictions match targets
        logits[:, :, :, 0] = 10.0

        # Ignore half the tokens
        targets[:, :2, :] = -100

        acc = compute_accuracy(logits, targets)
        assert torch.isclose(acc, torch.tensor(1.0))

    def test_compute_diff_accuracy(self):
        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 8, 8))
        buggy_ids = torch.randint(0, vocab_size, (2, 8, 8))

        result = compute_diff_accuracy(logits, targets, buggy_ids)

        assert "overall_accuracy" in result
        assert "changed_accuracy" in result
        assert "unchanged_accuracy" in result
        assert "n_changed" in result
        assert "n_unchanged" in result

        assert 0.0 <= result["overall_accuracy"].item() <= 1.0
        assert 0.0 <= result["changed_accuracy"].item() <= 1.0
        assert 0.0 <= result["unchanged_accuracy"].item() <= 1.0

    def test_compute_edit_distance_metrics_exact_match(self):
        pred_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        target_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

        metrics = compute_edit_distance_metrics(pred_ids, target_ids)

        assert torch.isclose(metrics["exact_match"], torch.tensor(1.0))
        assert torch.isclose(metrics["token_error_rate"], torch.tensor(0.0))
        assert torch.isclose(metrics["edit_count"], torch.tensor(0.0))

    def test_compute_edit_distance_metrics_partial_match(self):
        pred_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        target_ids = torch.tensor([[1, 2, 4], [4, 5, 6]])  # One mismatch in first seq

        metrics = compute_edit_distance_metrics(pred_ids, target_ids)

        # 50% exact match (1 of 2 sequences)
        assert torch.isclose(metrics["exact_match"], torch.tensor(0.5))
        # 1/6 tokens wrong on average
        assert metrics["token_error_rate"].item() > 0
        assert metrics["edit_count"].item() > 0

    def test_compute_edit_distance_metrics_with_ignore(self):
        pred_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        target_ids = torch.tensor([[-100, 2, 3], [4, 5, 6]])  # First token ignored

        metrics = compute_edit_distance_metrics(pred_ids, target_ids, ignore_index=-100)

        # Ignored tokens shouldn't count as errors
        assert torch.isclose(metrics["exact_match"], torch.tensor(1.0))

    def test_compute_sequence_accuracy_all_correct(self):
        vocab_size = 10
        logits = torch.zeros(2, 4, 4, vocab_size)
        targets = torch.zeros(2, 4, 4, dtype=torch.long)
        logits[:, :, :, 0] = 10.0

        acc = compute_sequence_accuracy(logits, targets)
        assert torch.isclose(acc, torch.tensor(1.0))

    def test_compute_sequence_accuracy_partial(self):
        vocab_size = 10
        logits = torch.zeros(2, 4, 4, vocab_size)
        targets = torch.zeros(2, 4, 4, dtype=torch.long)

        # First batch correct
        logits[0, :, :, 0] = 10.0
        # Second batch has one wrong
        logits[1, :, :, 0] = 10.0
        logits[1, 0, 0, 1] = 20.0  # This will be argmax but target is 0

        acc = compute_sequence_accuracy(logits, targets)
        assert torch.isclose(acc, torch.tensor(0.5))


class TestLossIntegration:
    """Integration tests for losses with model-like scenarios."""

    def test_training_loop_simulation(self):
        """Simulate a training loop to verify losses work end-to-end."""
        loss_fn = CombinedCodeRepairLoss(
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            n_iterations=4,
        )

        vocab_size = 100
        batch_size = 4
        height, width = 16, 12

        # Simulate model parameters
        model_param = nn.Parameter(torch.randn(vocab_size, 64))
        optimizer = torch.optim.Adam([model_param], lr=0.001)

        # Training step
        for step in range(3):
            optimizer.zero_grad()

            # Simulate model forward
            hidden = torch.randn(batch_size, height, width, 64)
            logits = torch.einsum("bhwd,vd->bhwv", hidden, model_param)

            # Generate targets and buggy ids
            targets = torch.randint(0, vocab_size, (batch_size, height, width))
            buggy_ids = torch.randint(0, vocab_size, (batch_size, height, width))

            # Simulate intermediate outputs
            all_logits = [
                torch.einsum("bhwd,vd->bhwv", torch.randn(batch_size, height, width, 64), model_param)
                for _ in range(4)
            ]

            # Compute loss
            result = loss_fn(
                logits,
                targets,
                buggy_ids=buggy_ids,
                all_iteration_logits=all_logits,
            )

            # Backward
            result["total"].backward()

            # Check gradients
            assert model_param.grad is not None
            assert not torch.isnan(model_param.grad).any()

            # Optimizer step
            optimizer.step()

    def test_loss_decreases_on_correct_predictions(self):
        """Test that loss is lower when predictions match targets."""
        loss_fn = TokenCrossEntropyLoss()

        vocab_size = 100
        targets = torch.randint(0, vocab_size, (2, 8, 8))

        # Random logits
        random_logits = torch.randn(2, 8, 8, vocab_size)

        # Logits that match targets (high confidence correct predictions)
        correct_logits = torch.zeros(2, 8, 8, vocab_size) - 10  # Low everywhere
        for b in range(2):
            for i in range(8):
                for j in range(8):
                    correct_logits[b, i, j, targets[b, i, j]] = 10.0

        loss_random = loss_fn(random_logits, targets)
        loss_correct = loss_fn(correct_logits, targets)

        assert loss_correct < loss_random

    def test_diff_loss_focuses_on_changes(self):
        """Test that diff loss properly weights changed tokens."""
        loss_fn = DiffWeightedLoss(diff_weight=10.0, base_weight=1.0)

        vocab_size = 100
        batch_size = 2
        height, width = 8, 8

        # Create scenario where changed tokens have high error
        targets = torch.randint(0, vocab_size, (batch_size, height, width))
        buggy_ids = targets.clone()

        # Mark first row as changed
        buggy_ids[:, 0, :] = (targets[:, 0, :] + 1) % vocab_size

        # Create logits that are wrong on changed tokens, right elsewhere
        logits = torch.zeros(batch_size, height, width, vocab_size) - 10
        for b in range(batch_size):
            for i in range(height):
                for j in range(width):
                    # Correct prediction everywhere except row 0
                    if i != 0:
                        logits[b, i, j, targets[b, i, j]] = 10.0
                    else:
                        # Wrong prediction on row 0
                        wrong_idx = (targets[b, i, j].item() + 1) % vocab_size
                        logits[b, i, j, wrong_idx] = 10.0

        diff_mask = loss_fn.compute_diff_mask(buggy_ids, targets)
        loss = loss_fn(logits, targets, diff_mask)

        # Loss should be high because changed tokens (row 0) have high weight and high error
        assert loss.item() > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_diff_mask(self):
        """Test when no tokens are changed."""
        loss_fn = DiffWeightedLoss(diff_weight=5.0)

        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 8, 8))
        diff_mask = torch.zeros(2, 8, 8, dtype=torch.bool)

        loss = loss_fn(logits, targets, diff_mask)
        assert not torch.isnan(loss)

    def test_all_tokens_changed(self):
        """Test when all tokens are changed."""
        loss_fn = DiffWeightedLoss(diff_weight=5.0)

        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 8, 8))
        diff_mask = torch.ones(2, 8, 8, dtype=torch.bool)

        loss = loss_fn(logits, targets, diff_mask)
        assert not torch.isnan(loss)

    def test_single_iteration(self):
        """Test intermediate supervision with single iteration."""
        loss_fn = IntermediateSupervisionLoss(n_iterations=4)

        vocab_size = 100
        all_logits = [torch.randn(2, 8, 8, vocab_size)]
        targets = torch.randint(0, vocab_size, (2, 8, 8))

        loss, info = loss_fn(all_logits, targets)

        assert info["n_iterations"] == 1
        assert not torch.isnan(loss)

    def test_all_ignored_tokens(self):
        """Test accuracy with all tokens ignored."""
        logits = torch.randn(2, 8, 8, 100)
        targets = torch.full((2, 8, 8), -100, dtype=torch.long)

        acc = compute_accuracy(logits, targets)
        # Should return 1.0 (perfect) when nothing to compare
        assert acc.item() == 1.0

    def test_very_large_logits(self):
        """Test numerical stability with large logits."""
        loss_fn = TokenCrossEntropyLoss()

        vocab_size = 100
        logits = torch.randn(2, 8, 8, vocab_size) * 100  # Large values
        targets = torch.randint(0, vocab_size, (2, 8, 8))

        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
