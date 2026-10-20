"""One independent objective/task per allocated GPU node, without DDP."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
tasks=json.loads(Path(sys.argv[1]).read_text())
def run_item(item,rank):
    env=dict(os.environ,LIBRI_OBJECTIVE=item['objective'],LIBRI_AVG_TASK=item['task'])
    print('INTERACTIVE_RANK_START',rank,item,os.uname().nodename,flush=True)
    code=subprocess.run(['bash',str(HERE/'run_one.pbs')],env=env).returncode
    print('INTERACTIVE_RANK_END',rank,item,code,flush=True)
    # A failed unit is audited by the driver; do not interrupt other ranks.
    (HERE/f"interactive_rank_{item['objective']}_{item['task']}_exit.txt").write_text(str(code)+'\n')

if len(sys.argv)>2 and sys.argv[2]=='--one-node':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda pair:run_item(pair[1],pair[0]),enumerate(tasks)))
else:
    rank=int(os.environ.get('OMPI_COMM_WORLD_RANK','0'))
    if rank<len(tasks):run_item(tasks[rank],rank)
