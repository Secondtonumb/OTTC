"""Zapformer OTTC upstream jsd, preserving the saved CE source and state keys."""
from types import FunctionType

from model_10ms import MFAZapformer as CEZapformer
from mfa_jsd.loss import batched_ottc_jsd_loss


class MFAZapformer(CEZapformer):
    loss_objective = 'jsd_upstream'

    def __init__(self, objective='ottc', tiny=False):
        if objective != 'ottc':
            raise ValueError('The upstream jsd entry supports the OTTC arm only')
        super().__init__('ottc', tiny=tiny)


# Reuse the exact existing forward, including 10 ms checks, control masks and
# trainable duration weights. Bind only its cost function in a PRIVATE globals
# dictionary; the CE class/module and running CE jobs are unaffected.
_forward = CEZapformer.forward
_globals = dict(_forward.__globals__)
_globals['batched_ottc_loss_bucketized'] = batched_ottc_jsd_loss
MFAZapformer.forward = FunctionType(_forward.__code__, _globals,
                                   'forward', _forward.__defaults__, _forward.__closure__)
MFAZapformer.forward.__doc__ = 'Original MFA OTTC forward with upstream jsd transport cost.'
