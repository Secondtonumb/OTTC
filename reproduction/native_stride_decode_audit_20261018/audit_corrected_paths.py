"""Check both fresh text-free and supplied-phone paths against original arcs."""
import ast
from collections import defaultdict
import json
from pathlib import Path
from audit_saved_paths import HERE, EXPERIMENT, VENDOR, source_helpers, accepted


def main():
    helpers, parity = source_helpers()
    names = {'normalize_repeat_separator', 'normalize_silence_self_loops', 'build_utterance_topology_arcs'}
    nodes = [n for n in ast.parse(VENDOR.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]))
    exec(compile(module,str(VENDOR),'exec'),helpers)
    results = []
    for cost in ['jsd_upstream', 'ce', 'jsd_standard']:
        name = 'wav2vec2__timit__10ms__s3t2ss__' + cost
        model = EXPERIMENT/'models'/name
        output = HERE/'results'/name
        assert (output/'CORRECTED_EVALUATION_COMPLETE.json').exists(), name
        vocab = json.loads((model/'vocab_used.json').read_text())
        config = json.loads((model/'run_config.json').read_text())
        mapping = {int(k):v for k,v in config['phone_to_substates'].items()}
        labels = config['topology_labels'];index = {label:i for i,label in enumerate(labels)}
        grammar = defaultdict(list)
        for line in (HERE/(name+'.token_fst.txt')).read_text().splitlines():
            p=line.split()
            if len(p)==4:grammar[(int(p[0]),index[p[2]])].append((int(p[1]),p[3]))
        for path in sorted(output.glob('benchmark_*.predictions.jsonl')):
            total=free_ok=fa_ok=0;ids=set()
            for line in path.open():
                item=json.loads(line);total+=1
                assert item['utt'] not in ids;ids.add(item['utt'])
                free=item['fsa_text_free'];fa=item['forced_alignment']
                assert 'failure' not in free and 'failure' not in fa
                assert accepted(free['frame_token_ids'],free['hypothesis'],grammar)
                free_ok+=1
                supplied=[vocab[p] for p,_,_ in item['reference']]
                assert fa['supplied_target_ids']==supplied
                assert [p for p,_,_ in fa['predicted_spans']]==[p for p,_,_ in item['reference']]
                _,arcs,_,final=helpers['build_utterance_topology_arcs'](supplied,mapping,'s3_t2_star_star',blank_index=0,
                    repeat_separator='optional_blank',silence_self_loops='none')
                transitions=defaultdict(set)
                for arc in arcs:transitions[(arc.src,arc.label)].add(arc.dst)
                states={0}
                for label in fa['frame_token_ids']:
                    states={target for state in states for target in transitions.get((state,label),())}
                    assert states,(path,item['utt'],'illegal supplied-phone arc')
                assert final in states,(path,item['utt'],'unfinished supplied-phone path')
                fa_ok+=1
            assert total==len(ids)==(400 if '_dev.' in path.name else 192)
            results.append(dict(recipe=name,predictions=str(path),utterances=total,valid_text_free_paths=free_ok,
                                valid_supplied_phone_paths=fa_ok))
    report=dict(utterances=sum(r['utterances'] for r in results),
                valid_text_free_paths=sum(r['valid_text_free_paths'] for r in results),
                valid_supplied_phone_paths=sum(r['valid_supplied_phone_paths'] for r in results),records=results)
    (HERE/'CORRECTED_PATH_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
