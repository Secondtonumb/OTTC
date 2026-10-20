"""Draw the actual frozen IF-MDD b-phone arcs using migrated inventory IDs."""
import json
import subprocess
from task_ottc import HERE, criterion
from utils.subphonetic_topology import token_fst_text

m=criterion()
symbolic=token_fst_text(['b'],m.topology,split_silence=False,blank_label='<blank>')
labels={label:i for i,label in enumerate(m.labels)}
lines=['digraph topology {', 'rankdir=LR;',
       'label="IF-MDD S3T2** phone b: acoustic state / phone output";',
       'labelloc=t;', 'node [shape=circle];',
       '0 [shape=doublecircle,label="phone boundary"];',
       'start [shape=point]; start -> 0;']
arcs=[]
for line in symbolic.splitlines():
    p=line.split()
    if len(p)==1:continue
    src,dst,ilabel,olabel=p
    caption=f'{ilabel} (ID {labels[ilabel]}) / {olabel}'
    lines.append(f'{src} -> {dst} [label={json.dumps(caption)}];')
    arcs.append(dict(src=int(src),dst=int(dst),state_id=labels[ilabel],state_label=ilabel,phone_output=olabel))
lines.append('}')
dot=HERE/'fsa_b.dot';dot.write_text('\n'.join(lines)+'\n')
for fmt in ['svg','pdf','png']:
    subprocess.run(['dot','-T'+fmt,str(dot),'-o',str(HERE/('fsa_b.'+fmt))],check=True)
(HERE/'fsa_b_arcs.json').write_text(json.dumps(arcs,indent=2)+'\n')
