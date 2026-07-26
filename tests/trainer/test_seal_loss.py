import torch

from verl.trainer.ppo.core_algos import seal_classification_loss


def test_classification_loss_separates_by_advantage_sign():
    advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    mask = torch.ones_like(advantages)
    # log_z aligned with advantage sign => low BCE
    aligned = torch.tensor([[5.0, 5.0, -5.0, -5.0]])
    # log_z anti-aligned => high BCE
    anti = torch.tensor([[-5.0, -5.0, 5.0, 5.0]])
    assert seal_classification_loss(aligned, advantages, mask) < seal_classification_loss(anti, advantages, mask)


def test_mask_excludes_tokens():
    advantages = torch.tensor([[1.0, -1.0]])
    log_z = torch.tensor([[5.0, 5.0]])  # 2nd token wrong (positive logit, negative adv)
    full = seal_classification_loss(log_z, advantages, torch.ones_like(advantages))
    masked = seal_classification_loss(log_z, advantages, torch.tensor([[1.0, 0.0]]))
    assert masked < full  # dropping the wrong token lowers the loss


def test_handles_trailing_singleton_dim():
    advantages = torch.tensor([[1.0, -1.0]])
    log_z = torch.tensor([[[2.0], [-2.0]]])  # shape (1, 2, 1) -> squeezed
    loss = seal_classification_loss(log_z, advantages, torch.ones_like(advantages))
    assert torch.isfinite(loss)
