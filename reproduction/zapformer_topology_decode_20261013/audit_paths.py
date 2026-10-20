"""Independent reconstruction of phones from complete legal state paths."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def reconstruct(path):
    output = []
    active = None
    previous_stage = -1
    for state_id in path:
        if state_id == 0:
            assert active is None or previous_stage == 2
            active = None
            previous_stage = -1
            continue
        phone, stage = (state_id - 1) // 3 + 1, (state_id - 1) % 3
        if active is None or phone != active or (previous_stage == 2 and stage == 0):
            assert active is None or previous_stage == 2
            assert stage == 0
            output.append(phone)
            active = phone
        else:
            assert stage >= previous_stage
        previous_stage = stage
    assert active is None or previous_stage == 2
    return output


def main():
    assert reconstruct([7, 7, 8, 8, 9, 9]) == [3]
    assert reconstruct([7, 7, 9, 9]) == [3]
    assert reconstruct([7, 9, 7, 9]) == [3, 3]
    assert reconstruct([7, 9, 0, 7, 9]) == [3, 3]
    count = 0
    counts = {}
    for file in sorted(HERE.glob('*/*_predictions.jsonl')):
        n = 0
        for r in map(json.loads, file.open()):
            assert reconstruct(r['legal_state_path']) == r['all_phone_ids'], (file, r['id'])
            n += 1
        counts[str(file.relative_to(HERE))] = n
        count += n
    result = dict(status='passed', sentences_checked=count,
                  independent_path_reconstruction_matches_fsa_phone_outputs=True,
                  repeated_self_loop_states_emit_one_phone=True,
                  state2_to_state0_is_new_phone_occurrence=True, files=counts)
    if (HERE / 'COMPLETE.json').exists():
        assert len(counts) == 16 and count == 4 * 2 * (400 + 192)
        result['full_suite_complete'] = True
    (HERE / 'SELF_LOOP_COLLAPSE_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
