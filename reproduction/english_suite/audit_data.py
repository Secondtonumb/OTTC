"""Audit complete split counts, valid targets, audio headers and native frame capacity."""
import json
from pathlib import Path
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {}
    for dataset, expected_train in [('libri100', 28538), ('libri960', 281191)]:
        directory = ROOT/f'multihead_{dataset}'/'data'
        vocabs = json.loads((directory/'vocabs.json').read_text())
        summary = {}
        all_ids = set()
        for split, expected in [('train-train', expected_train), ('train-dev', 5567), ('test', 5558)]:
            count = 0
            seconds = 0.
            for line in (directory/f'{split}.jsonl').open():
                e = json.loads(line)
                assert e['utt'] not in all_ids
                all_ids.add(e['utt'])
                for h, targets in e['targets'].items():
                    assert all(0 < t < len(vocabs[h]) for t in targets)
                    frames = (round(e['duration']*16000)-400)//320+1
                    assert len(targets)+sum(a == b for a, b in zip(targets, targets[1:])) <= frames, (e['utt'], h)
                if count % 5000 == 0:
                    info = sf.info(e['wav'])
                    assert info.samplerate == 16000 and info.channels == 1
                    assert abs(info.duration-e['duration']) < .02
                count += 1
                seconds += e['duration']
            assert count == expected, (dataset, split, count, expected)
            summary[split] = {'utterances': count, 'hours': seconds/3600}
        report[dataset] = summary
    output = ROOT/'english_suite'/'data_audit.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
