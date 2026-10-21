"""Optional Trainer logging for the temporal-smoothing OTTC model extension."""
from transformers import Trainer


class TemporalSmoothingTrainerMixin:
    """Merge component metrics before HF writes local history and W&B logs.

    Also works before a custom Trainer in the MRO, including trainers which
    call model(**microbatch).loss directly in training_step. Trainer.log is
    called on all ranks; reduce sums before dividing by total example count.
    Existing scalar loss/optimizer/epoch/LR logging is retained.
    """
    def log(self, logs, *args, **kwargs):
        if 'loss' in logs or 'train_loss' in logs:
            model = self.accelerator.unwrap_model(self.model)
            pop = getattr(model, 'pop_temporal_smoothing_statistics', None)
            values = pop() if pop is not None else None
            if values is not None:
                values = self.accelerator.reduce(values, reduction='sum')
                numbers = (values[:4] / values[4].clamp_min(1)).cpu().tolist()
                names = ['loss_ottc_per_utterance', 'loss_temporal_per_utterance',
                         'loss_temporal_weighted_per_utterance', 'loss_total_per_utterance']
                logs = dict(logs, **dict(zip(names, numbers)),
                    temporal_smoothing_weight=model.config.temporal_smoothing_weight,
                    temporal_smoothing_tau=model.config.temporal_smoothing_tau)
        return super().log(logs, *args, **kwargs)


class TemporalSmoothingTrainer(TemporalSmoothingTrainerMixin, Trainer):
    """HF Trainer with local/W&B component scalar logging; no upload side effects."""
