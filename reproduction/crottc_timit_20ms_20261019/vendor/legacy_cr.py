"""Frozen IF-MDD CR method; see provenance.json for source identity."""
import torch

class LegacyCR:
    def _compute_cr_loss(self, p_ctc_1, p_ctc_2, wav_lens, time_mask=None):
            """Compute Consistency Regularization loss for CR-CTC (GPU-optimized)"""
            B, T, C = p_ctc_1.shape
            device = p_ctc_1.device
            
            # Symmetric KL divergence
            kl_1_to_2 = torch.nn.functional.kl_div(
                input=p_ctc_1, target=p_ctc_2.detach(),
                reduction="none", log_target=True
            )
            kl_2_to_1 = torch.nn.functional.kl_div(
                input=p_ctc_2, target=p_ctc_1.detach(),
                reduction="none", log_target=True
            )
            cr_loss = kl_1_to_2 + kl_2_to_1  # [B, T, C]
            
            # Length mask to ignore padding (optimized)
            abs_lens = (wav_lens * T).long()
            # Create arange once and reuse
            time_indices = torch.arange(T, device=device, dtype=abs_lens.dtype).unsqueeze(0)
            length_mask = (time_indices >= abs_lens.unsqueeze(1)).unsqueeze(-1)
            
            cr_loss = cr_loss.masked_fill(length_mask, 0.0)
            
            # Average over valid positions (avoid creating large intermediate tensors)
            num_valid = (T * C * B) - length_mask.sum()
            cr_loss = cr_loss.sum() / torch.clamp(num_valid, min=1)
            
            return cr_loss
