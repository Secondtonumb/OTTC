"""Pause at an optimizer boundary, preserving the planned 40-epoch schedule."""
import time
from transformers import TrainerCallback


class TimeBudgetCallback(TrainerCallback):
    def __init__(self,deadline):self.deadline=deadline

    def on_step_end(self,args,state,control,**kwargs):
        if self.deadline is not None and time.monotonic()>=self.deadline:
            control.should_training_stop=True
            control.should_save=True
        return control
