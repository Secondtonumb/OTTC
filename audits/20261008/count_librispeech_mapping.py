"""Stream existing IF-MDD LibriSpeech manifests; count monophone targets."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/m64000/work/IF-MDD')
FOCUS = ['ao', 'zh', 'q', 'dx', 'pcl', 'tcl', 'kcl', 'bcl', 'dcl', 'gcl']


def entries(path):
    # Existing exporters indent top-level entries by exactly two spaces.
    # Decode each complete entry independently to keep memory bounded.
    block = []
    with path.open() as f:
        assert f.readline().strip() == '{'
        for line in f:
            if not block:
                if line.strip() == '}':
                    break
                assert line.startswith('  "') and line.rstrip().endswith(': {'), line[:100]
            block.append(line)
            if line.rstrip() in ('  },', '  }'):
                text = ''.join(block).rstrip().removesuffix(',')
                d = json.loads('{'+text+'}')
                assert len(d) == 1
                yield next(iter(d.values()))
                block = []
    assert not block


def summarize(path):
    target, spans, duration, affected = Counter(), Counter(), Counter(), Counter()
    n, audio_seconds, annotated_seconds, over20 = 0, 0, 0, 0
    over20_targets = Counter()
    mismatched_fields = 0
    aozh_utterances = 0
    repeated_targets = 0
    for e in entries(path):
        n += 1
        audio_seconds += float(e['duration'])
        tokens = e['perceived_train_target'].split()
        assert all('~' not in p and '_' not in p for p in tokens), 'Expected monophone data'
        target.update(tokens)
        affected.update(set(tokens))
        aozh_utterances += bool({'ao','zh'} & set(tokens))
        repeated_targets += sum(a == b for a,b in zip(tokens,tokens[1:]))
        mismatched_fields += e['perceived_train_target'] != e['canonical_aligned']
        if float(e['duration']) > 20:
            over20 += 1
            over20_targets.update(tokens)
        for s in e['phonemes']:
            p = s['phoneme']
            length = float(s['end']) - float(s['start'])
            assert length >= -1e-9
            spans[p] += 1
            duration[p] += length
            annotated_seconds += length
    nt, ns = sum(target.values()), sum(spans.values())
    phones = {}
    for p in sorted(set(target) | set(spans) | set(FOCUS)):
        phones[p] = {'target_count': target[p], 'target_pct': 100*target[p]/nt,
                     'target_nosil_pct': 100*target[p]/(nt-target['sil']) if p != 'sil' else None,
                     'interval_count': spans[p], 'interval_pct': 100*spans[p]/ns,
                     'duration_seconds': duration[p], 'annotated_duration_pct': 100*duration[p]/annotated_seconds,
                     'affected_utterances': affected[p], 'utterance_pct': 100*affected[p]/n}
    return {'source': str(path), 'utterances': n, 'audio_hours': audio_seconds/3600,
            'target_tokens_including_sil': nt, 'target_tokens_excluding_sil': nt-target['sil'],
            'phone_intervals': ns, 'annotated_seconds': annotated_seconds, 'phones': phones,
            'ao_zh_affected_utterances': aozh_utterances,
            'adjacent_same_target_pairs': repeated_targets,
            'canonical_perceived_target_mismatch_utterances': mismatched_fields,
            'duration_gt20_utterances': over20, 'duration_gt20_target_counts': dict(over20_targets)}


files = {
    'libri100_train': ROOT/'data/librispeech_context_phone/monophone/train-train.json',
    'libri360_train': ROOT/'data/librispeech_alignments_960/json/train_clean_360.json',
    'libri500_train': ROOT/'data/librispeech_alignments_960/json/train_other_500.json',
    'test_clean': ROOT/'data/librispeech_context_phone/monophone/librispeech_test_clean.json',
    'test_other': ROOT/'data/librispeech_context_phone/monophone/librispeech_test_other.json',
}
sets = {}
for name,path in files.items():
    sets[name] = summarize(path)
    d = sets[name]
    print(name, 'utterances', d['utterances'], 'target tokens', d['target_tokens_including_sil'],
          'ao',d['phones']['ao']['target_count'],'zh',d['phones']['zh']['target_count'], flush=True)

def combine(names):
    data = [sets[n] for n in names]
    addkeys = ['utterances','audio_hours','target_tokens_including_sil','target_tokens_excluding_sil',
               'phone_intervals','annotated_seconds','ao_zh_affected_utterances','adjacent_same_target_pairs',
               'canonical_perceived_target_mismatch_utterances','duration_gt20_utterances']
    d={k:sum(x[k] for x in data) for k in addkeys}
    d['sources'] = [x['source'] for x in data]
    d['phones']={}
    dropped = Counter()
    for x in data:
        dropped.update(x['duration_gt20_target_counts'])
    d['duration_gt20_target_counts']=dict(dropped)
    for p in sorted({p for x in data for p in x['phones']}):
        s={k:sum(x['phones'].get(p,{}).get(k,0) for x in data)
           for k in ['target_count','interval_count','duration_seconds','affected_utterances']}
        s.update(target_pct=100*s['target_count']/d['target_tokens_including_sil'],
                 target_nosil_pct=100*s['target_count']/d['target_tokens_excluding_sil'] if p != 'sil' else None,
                 interval_pct=100*s['interval_count']/d['phone_intervals'],
                 annotated_duration_pct=100*s['duration_seconds']/d['annotated_seconds'],
                 utterance_pct=100*s['affected_utterances']/d['utterances'])
        d['phones'][p]=s
    return d

sets['libri960_train']=combine(['libri100_train','libri360_train','libri500_train'])
sets['test_combined']=combine(['test_clean','test_other'])
assert sets['libri100_train']['utterances']==28538
assert sets['libri960_train']['utterances']==281191
assert sets['test_combined']['utterances']==5558
provenance={'target_field':'perceived_train_target',
            'target_denominator':'All saved monophone training/evaluation target tokens, including sil',
            'alignment_source':'gilkeyio/librispeech-alignments as exported by IF-MDD',
            'not_manual_timit_phn':True,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'librispeech_mapping_counts.json').write_text(json.dumps({'provenance':provenance,'sets':sets},indent=2)+'\n')
with (OUT/'librispeech_mapping_counts.csv').open('w') as f:
    writer=csv.writer(f)
    writer.writerow(['dataset','phone','target_count','target_pct_including_sil','target_pct_excluding_sil','interval_count','duration_seconds','affected_utterances'])
    for name,d in sets.items():
        for p,s in d['phones'].items():
            writer.writerow([name,p]+[s[k] for k in ['target_count','target_pct','target_nosil_pct','interval_count','duration_seconds','affected_utterances']])
lines=['# IF-MDD LibriSpeech 音素映射数量审计','',
       '统计实际保存的 monophone perceived_train_target，每次出现计一个 token；包含 sil。另存 no-sil 比例、对齐区间数量及时长。',
       '来源为旧 IF-MDD 的 gilkeyio/librispeech-alignments 导出数据，不是 TIMIT 原始人工 PHN 标注。不存在的闭塞标签不代表录音中不存在闭塞。',
       '原始 OTTC README 的 LibriSpeech 实验是文本字符识别；此处是用户之前 IF-MDD 的音素实验数据。','',
       '| 数据集 | 语音数 | 目标 token 总数（含 sil） | ao 个数（比例） | zh 个数（比例） | ao+zh 比例 | q/dx/六类闭塞标签 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for name in ['libri100_train','libri960_train','test_clean','test_other','test_combined']:
    d=sets[name]; a=d['phones']['ao'];z=d['phones']['zh']
    absent=sum(d['phones'][p]['target_count'] for p in FOCUS[2:])
    lines += [f"| {name} | {d['utterances']:,} | {d['target_tokens_including_sil']:,} | {a['target_count']:,} ({a['target_pct']:.2f}%) | {z['target_count']:,} ({z['target_pct']:.2f}%) | {a['target_pct']+z['target_pct']:.2f}% | {absent} |"]
lines += ['', '## 时长过滤与比例解释', '']
for name in ['libri100_train','libri960_train']:
    d=sets[name];drop=d['duration_gt20_target_counts'];denom=d['target_tokens_including_sil']-sum(drop.values())
    focus=d['phones']['ao']['target_count']+d['phones']['zh']['target_count']
    nosil=100*focus/d['target_tokens_excluding_sil']
    filtered=100*(focus-drop.get('ao',0)-drop.get('zh',0))/denom
    lines += [f"- {name}：ao+zh 的 no-sil 比例 {nosil:.2f}%；若过滤 duration >20s 的语音，移除 {d['duration_gt20_utterances']} 条，ao+zh 的含 sil 比例为 {filtered:.2f}%。"]
lines += ['', '训练 manifest 的 canonical_aligned 与 perceived_train_target 一致性已逐条统计；详细 JSON 提供全部音素及各训练子集。这些频次不能直接换算成 PER 或 segmentation F1 的变化。']
(OUT/'LIBRISPEECH_MAPPING_COUNTS_ZH.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
