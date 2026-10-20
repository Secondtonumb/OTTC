"""Continue the archived native schedule at its final LR without a restart."""
def continuation_scheduler(native_scheduler, original_epochs):
    class ContinuationScheduler(native_scheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tot_batches = sum(self.batches_per_epoch[:original_epochs])

        def get_progress(self):
            if self.epoch > original_epochs:
                return 1.0
            return super().get_progress()

    return ContinuationScheduler
