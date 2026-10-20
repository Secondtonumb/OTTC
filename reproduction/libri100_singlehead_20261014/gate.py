"""Require finite gradients and a real K2 evaluation for all three costs."""
import bootstrap
from bootstrap import HERE
import json
from train import hashes, OBJECTIVES

records={}
for objective in OBJECTIVES:
    training=json.loads((HERE/'validation'/objective/'PASS.json').read_text())
    evaluation=json.loads((HERE/'validation_evaluation'/objective/'EVAL_COMPLETE.json').read_text())
    assert training['source_sha256']==hashes()
    assert training['batch64_longest_inputs_safe'] and training['freeze_transition_tested']
    assert evaluation['smoke'] and evaluation['utterances']=={'dev_clean':8}
    assert evaluation['failures']=={'dev_clean':0}
    records[objective]=dict(training=training,evaluation=evaluation)
(HERE/'GPU_VALIDATION_COMPLETE.json').write_text(json.dumps(records,indent=2)+'\n')
print('ALL_THREE_GPU_K2_VALIDATED',flush=True)
