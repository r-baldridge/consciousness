"""
TRM Loss Functions for Code Repair Tasks

This module implements specialized loss functions for training the TRM model
on code repair tasks, with focus on buggy-to-fixed code transformation.

Loss Components:
- TokenCrossEntropyLoss: Standard cross-entropy with label smoothing
- DiffWeightedLoss: Higher weight on tokens that differ (bug locations)
- IntermediateSupervisionLoss: Loss at each recursive iteration
- CombinedCodeRepairLoss: Unified loss combining all components

Utility Functions:
- compute_perplexity: Compute perplexity from loss
- compute_accuracy: Token-level accuracy with ignore index
- compute_edit_distance_metrics: Edit distance between predictions and targets

Reference:
    "Less is More: Recursive Reasoning with Tiny Networks"
    https://arxiv.org/abs/2510.04871
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for next token prediction.

    Standard cross-entropy loss with support for label smoothing,
    padding token handling, and multiple reduction options.

    Args:
        label_smoothing: Label smoothing factor (0.0 to 1.0).
                        Smooths target distribution to prevent overconfidence.
        ignore_index: Token index to ignore in loss computation (e.g., padding).
        reduction: How to reduce the loss: 'mean', 'sum', or 'none'.

    Example:
        >>> loss_fn = TokenCrossEntropyLoss(label_smoothing=0.1, ignore_index=0)
        >>> logits = torch.randn(2, 64, 48, 1000)  # batch, height, width, vocab
        >>> targets = torch.randint(0, 1000, (2, 64, 48))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}. Use 'mean', 'sum', or 'none'.")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Predicted logits of shape (..., vocab_size).
            targets: Target token indices of shape (...).

        Returns:
            Loss tensor (scalar if reduction='mean'/'sum', otherwise same shape as targets).
        """
        # Get vocab size from logits
        vocab_size = logits.shape[-1]

        # Flatten logits and targets for cross_entropy
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute cross-entropy with label smoothing
        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )

        return loss


class DiffWeightedLoss(nn.Module):
    """Cross-entropy loss with higher weight on changed tokens.

    Focuses training on tokens that differ between buggy and fixed code,
    making the model pay more attention to actual bug repairs.

    Args:
        diff_weight: Weight multiplier for tokens that changed (default: 5.0).
        base_weight: Weight for unchanged tokens (default: 1.0).
        label_smoothing: Label smoothing factor.
        ignore_index: Token index to ignore.

    Example:
        >>> loss_fn = DiffWeightedLoss(diff_weight=5.0)
        >>> buggy_ids = torch.randint(0, 100, (2, 64, 48))
        >>> fixed_ids = torch.randint(0, 100, (2, 64, 48))
        >>> diff_mask = loss_fn.compute_diff_mask(buggy_ids, fixed_ids)
        >>> logits = torch.randn(2, 64, 48, 100)
        >>> loss = loss_fn(logits, fixed_ids, diff_mask)
    """

    def __init__(
        self,
        diff_weight: float = 5.0,
        base_weight: float = 1.0,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.diff_weight = diff_weight
        self.base_weight = base_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def compute_diff_mask(
        self,
        buggy_ids: torch.Tensor,
        fixed_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mask indicating which tokens differ between buggy and fixed code.

        Args:
            buggy_ids: Buggy code token IDs of shape (...).
            fixed_ids: Fixed code token IDs of shape (...).

        Returns:
            Boolean tensor where True indicates a changed token.
        """
        return buggy_ids != fixed_ids

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        diff_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute diff-weighted cross-entropy loss.

        Args:
            logits: Predicted logits of shape (..., vocab_size).
            targets: Target token indices of shape (...).
            diff_mask: Boolean mask where True = changed token.
                      If None, all tokens are weighted equally.

        Returns:
            Weighted loss scalar.
        """
        vocab_size = logits.shape[-1]

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute per-token losses
        per_token_loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Create weight tensor
        if diff_mask is not None:
            diff_mask_flat = diff_mask.reshape(-1)
            weights = torch.where(
                diff_mask_flat,
                torch.full_like(per_token_loss, self.diff_weight),
                torch.full_like(per_token_loss, self.base_weight),
            )
        else:
            weights = torch.ones_like(per_token_loss)

        # Zero out weights for ignored tokens
        valid_mask = targets_flat != self.ignore_index
        weights = weights * valid_mask.float()

        # Compute weighted mean
        weighted_loss = (per_token_loss * weights).sum()
        weight_sum = weights.sum()

        if weight_sum > 0:
            return weighted_loss / weight_sum
        return weighted_loss


class IntermediateSupervisionLoss(nn.Module):
    """Loss computed at each recursive iteration with weighted combination.

    Provides supervision signal at each step of the recursive computation,
    with later iterations typically weighted more heavily.

    Args:
        n_iterations: Expected number of iterations.
        iteration_weights: List of weights for each iteration, or 'linear' or 'exponential'.
                          If None, uses uniform weighting.
        learnable_weights: If True, weights are learned during training.
        label_smoothing: Label smoothing factor.
        ignore_index: Token index to ignore.

    Example:
        >>> loss_fn = IntermediateSupervisionLoss(n_iterations=8, iteration_weights='exponential')
        >>> all_logits = [torch.randn(2, 64, 48, 100) for _ in range(8)]
        >>> targets = torch.randint(0, 100, (2, 64, 48))
        >>> loss, info = loss_fn(all_logits, targets)
    """

    def __init__(
        self,
        n_iterations: int = 8,
        iteration_weights: Optional[Union[List[float], str]] = None,
        learnable_weights: bool = False,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.n_iterations = n_iterations
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.learnable_weights = learnable_weights

        # Initialize iteration weights
        if iteration_weights is None:
            weights = [1.0] * n_iterations
        elif isinstance(iteration_weights, str):
            if iteration_weights == "linear":
                # Linear increase: 1, 2, 3, ...
                weights = [(i + 1) for i in range(n_iterations)]
            elif iteration_weights == "exponential":
                # Exponential increase: 2^0, 2^1, 2^2, ...
                weights = [2.0 ** i for i in range(n_iterations)]
            else:
                raise ValueError(f"Unknown weight scheme: {iteration_weights}")
        else:
            weights = list(iteration_weights)

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

        if learnable_weights:
            # Use log-space for stability
            self.log_weights = nn.Parameter(
                torch.tensor([math.log(w + 1e-8) for w in weights])
            )
        else:
            self.register_buffer(
                "weights",
                torch.tensor(weights, dtype=torch.float32),
            )

    def get_weights(self) -> torch.Tensor:
        """Get current iteration weights (normalized to sum to 1)."""
        if self.learnable_weights:
            weights = F.softmax(self.log_weights, dim=0)
        else:
            weights = self.weights
        return weights

    def forward(
        self,
        all_iteration_logits: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute weighted loss across all iterations.

        Args:
            all_iteration_logits: List of logits from each iteration.
                                 Each has shape (..., vocab_size).
            targets: Target token indices of shape (...).

        Returns:
            Tuple of (total_loss, info_dict):
                - total_loss: Weighted sum of iteration losses
                - info_dict: Dictionary with per-iteration losses and weights
        """
        n_actual = len(all_iteration_logits)
        weights = self.get_weights()

        # Handle case where actual iterations differ from expected
        if n_actual < len(weights):
            weights = weights[:n_actual]
            # Renormalize
            weights = weights / weights.sum()

        vocab_size = all_iteration_logits[0].shape[-1]
        targets_flat = targets.reshape(-1)

        iteration_losses = []
        weighted_sum = torch.tensor(0.0, device=targets.device)

        for i, logits in enumerate(all_iteration_logits):
            logits_flat = logits.reshape(-1, vocab_size)

            # Compute loss for this iteration
            loss_i = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.ignore_index,
                label_smoothing=self.label_smoothing,
            )
            iteration_losses.append(loss_i)

            # Accumulate weighted loss
            if i < len(weights):
                weighted_sum = weighted_sum + weights[i] * loss_i

        info = {
            "iteration_losses": torch.stack(iteration_losses),
            "iteration_weights": weights[:n_actual],
            "n_iterations": n_actual,
        }

        return weighted_sum, info


class CombinedCodeRepairLoss(nn.Module):
    """Combined loss for code repair training.

    Combines multiple loss components with configurable weights:
    - Token cross-entropy loss (alpha)
    - Diff-weighted loss (beta)
    - Intermediate supervision loss (gamma)

    Args:
        alpha: Weight for base cross-entropy loss.
        beta: Weight for diff-weighted loss.
        gamma: Weight for intermediate supervision loss.
        diff_weight: Weight multiplier for changed tokens.
        n_iterations: Expected number of recursion iterations.
        iteration_weights: Weighting scheme for intermediate supervision.
        label_smoothing: Label smoothing factor.
        ignore_index: Token index to ignore.
        gradient_scales: Optional dict to scale gradients per component.

    Example:
        >>> loss_fn = CombinedCodeRepairLoss(alpha=0.5, beta=0.3, gamma=0.2)
        >>> logits = torch.randn(2, 64, 48, 100)
        >>> targets = torch.randint(0, 100, (2, 64, 48))
        >>> buggy_ids = torch.randint(0, 100, (2, 64, 48))
        >>> all_logits = [logits]  # Single iteration
        >>> loss_dict = loss_fn(logits, targets, buggy_ids=buggy_ids, all_iteration_logits=all_logits)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        diff_weight: float = 5.0,
        n_iterations: int = 8,
        iteration_weights: Optional[Union[List[float], str]] = "exponential",
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        gradient_scales: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gradient_scales = gradient_scales or {}

        # Component losses
        self.token_ce_loss = TokenCrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

        self.diff_loss = DiffWeightedLoss(
            diff_weight=diff_weight,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

        self.intermediate_loss = IntermediateSupervisionLoss(
            n_iterations=n_iterations,
            iteration_weights=iteration_weights,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )

    def _apply_gradient_scale(
        self,
        loss: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """Apply gradient scaling to a loss component."""
        if name not in self.gradient_scales:
            return loss

        scale = self.gradient_scales[name]

        # Use gradient scaling via custom autograd
        # Forward: loss unchanged, Backward: gradient * scale
        return GradientScale.apply(loss, scale)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        buggy_ids: Optional[torch.Tensor] = None,
        all_iteration_logits: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Final predicted logits (..., vocab_size).
            targets: Target token indices (...).
            buggy_ids: Original buggy code token IDs (for diff mask).
            all_iteration_logits: Logits from each iteration (for intermediate supervision).

        Returns:
            Dictionary containing:
                - 'total': Combined total loss
                - 'ce_loss': Base cross-entropy loss
                - 'diff_loss': Diff-weighted loss (if buggy_ids provided)
                - 'intermediate_loss': Intermediate supervision loss (if all_iteration_logits provided)
                - 'iteration_losses': Per-iteration losses (if applicable)
        """
        result = {}

        # 1. Base cross-entropy loss
        ce_loss = self.token_ce_loss(logits, targets)
        ce_loss = self._apply_gradient_scale(ce_loss, "ce_loss")
        result["ce_loss"] = ce_loss

        total_loss = self.alpha * ce_loss

        # 2. Diff-weighted loss (if buggy_ids provided)
        if buggy_ids is not None and self.beta > 0:
            diff_mask = self.diff_loss.compute_diff_mask(buggy_ids, targets)
            diff_loss = self.diff_loss(logits, targets, diff_mask)
            diff_loss = self._apply_gradient_scale(diff_loss, "diff_loss")
            result["diff_loss"] = diff_loss
            result["diff_mask_sum"] = diff_mask.sum().float()
            total_loss = total_loss + self.beta * diff_loss

        # 3. Intermediate supervision loss (if all_iteration_logits provided)
        if all_iteration_logits is not None and self.gamma > 0:
            intermediate_loss, inter_info = self.intermediate_loss(
                all_iteration_logits, targets
            )
            intermediate_loss = self._apply_gradient_scale(
                intermediate_loss, "intermediate_loss"
            )
            result["intermediate_loss"] = intermediate_loss
            result["iteration_losses"] = inter_info["iteration_losses"]
            result["iteration_weights"] = inter_info["iteration_weights"]
            total_loss = total_loss + self.gamma * intermediate_loss

        result["total"] = total_loss

        return result


class GradientScale(torch.autograd.Function):
    """Custom autograd function for gradient scaling.

    Keeps forward pass unchanged but scales gradients during backprop.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output * ctx.scale, None


# =============================================================================
# Utility Functions
# =============================================================================


def compute_perplexity(loss: torch.Tensor) -> torch.Tensor:
    """
    Compute perplexity from cross-entropy loss.

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value (scalar tensor).

    Returns:
        Perplexity value (scalar tensor).

    Example:
        >>> loss = torch.tensor(2.0)
        >>> ppl = compute_perplexity(loss)
        >>> print(f"Perplexity: {ppl.item():.2f}")  # ~7.39
    """
    return torch.exp(loss)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute token-level accuracy.

    Args:
        logits: Predicted logits of shape (..., vocab_size).
        targets: Target token indices of shape (...).
        ignore_index: Token index to exclude from accuracy computation.

    Returns:
        Accuracy as a scalar tensor (0.0 to 1.0).

    Example:
        >>> logits = torch.randn(2, 64, 48, 100)
        >>> targets = torch.randint(0, 100, (2, 64, 48))
        >>> acc = compute_accuracy(logits, targets)
    """
    predictions = logits.argmax(dim=-1)

    # Create mask for valid tokens
    valid_mask = targets != ignore_index

    # Count correct predictions
    correct = (predictions == targets) & valid_mask
    total_valid = valid_mask.sum()

    if total_valid == 0:
        return torch.tensor(1.0, device=logits.device)

    accuracy = correct.sum().float() / total_valid.float()
    return accuracy


def compute_diff_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    buggy_ids: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Compute accuracy separately for changed and unchanged tokens.

    Args:
        logits: Predicted logits of shape (..., vocab_size).
        targets: Target token indices of shape (...).
        buggy_ids: Original buggy code token IDs.
        ignore_index: Token index to exclude.

    Returns:
        Dictionary with:
            - 'overall_accuracy': Total accuracy
            - 'changed_accuracy': Accuracy on changed tokens
            - 'unchanged_accuracy': Accuracy on unchanged tokens
            - 'n_changed': Number of changed tokens
            - 'n_unchanged': Number of unchanged tokens
    """
    predictions = logits.argmax(dim=-1)

    # Valid mask
    valid_mask = targets != ignore_index

    # Diff mask
    diff_mask = buggy_ids != targets

    # Overall accuracy
    correct = (predictions == targets) & valid_mask
    overall_acc = correct.sum().float() / valid_mask.sum().float()

    # Changed tokens accuracy
    changed_valid = valid_mask & diff_mask
    n_changed = changed_valid.sum()
    if n_changed > 0:
        changed_correct = correct & diff_mask
        changed_acc = changed_correct.sum().float() / n_changed.float()
    else:
        changed_acc = torch.tensor(1.0, device=logits.device)

    # Unchanged tokens accuracy
    unchanged_valid = valid_mask & ~diff_mask
    n_unchanged = unchanged_valid.sum()
    if n_unchanged > 0:
        unchanged_correct = correct & ~diff_mask
        unchanged_acc = unchanged_correct.sum().float() / n_unchanged.float()
    else:
        unchanged_acc = torch.tensor(1.0, device=logits.device)

    return {
        "overall_accuracy": overall_acc,
        "changed_accuracy": changed_acc,
        "unchanged_accuracy": unchanged_acc,
        "n_changed": n_changed,
        "n_unchanged": n_unchanged,
    }


def compute_edit_distance_metrics(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Compute edit distance related metrics between predictions and targets.

    Computes Levenshtein-like metrics at the sequence level.

    Args:
        pred_ids: Predicted token IDs of shape (batch, ...).
        target_ids: Target token IDs of shape (batch, ...).
        ignore_index: Token index to exclude.

    Returns:
        Dictionary with:
            - 'exact_match': Fraction of sequences that match exactly
            - 'token_error_rate': Average fraction of tokens that differ
            - 'edit_count': Average number of different tokens per sequence

    Example:
        >>> pred_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> target_ids = torch.tensor([[1, 2, 4], [4, 5, 6]])
        >>> metrics = compute_edit_distance_metrics(pred_ids, target_ids)
    """
    batch_size = pred_ids.shape[0]

    # Flatten spatial dimensions
    pred_flat = pred_ids.reshape(batch_size, -1)
    target_flat = target_ids.reshape(batch_size, -1)

    # Valid mask per sequence
    valid_mask = target_flat != ignore_index

    # Token differences per sequence
    diff = (pred_flat != target_flat) & valid_mask
    diff_count_per_seq = diff.sum(dim=-1).float()
    valid_count_per_seq = valid_mask.sum(dim=-1).float()

    # Exact match: sequences with zero differences
    exact_match = (diff_count_per_seq == 0).float().mean()

    # Token error rate: average fraction of errors
    token_error_rate = (diff_count_per_seq / valid_count_per_seq.clamp(min=1)).mean()

    # Average edit count
    edit_count = diff_count_per_seq.mean()

    return {
        "exact_match": exact_match,
        "token_error_rate": token_error_rate,
        "edit_count": edit_count,
    }


def compute_sequence_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute sequence-level accuracy (all tokens correct).

    Args:
        logits: Predicted logits of shape (batch, ..., vocab_size).
        targets: Target token indices of shape (batch, ...).
        ignore_index: Token index to exclude.

    Returns:
        Fraction of sequences with all tokens correct.
    """
    predictions = logits.argmax(dim=-1)
    batch_size = predictions.shape[0]

    # Flatten spatial dimensions
    pred_flat = predictions.reshape(batch_size, -1)
    target_flat = targets.reshape(batch_size, -1)

    # Valid mask
    valid_mask = target_flat != ignore_index

    # Check if all valid tokens match per sequence
    # A token matches if either it's correct OR it's ignored
    matches = (pred_flat == target_flat) | ~valid_mask
    all_correct = matches.all(dim=-1)

    return all_correct.float().mean()
