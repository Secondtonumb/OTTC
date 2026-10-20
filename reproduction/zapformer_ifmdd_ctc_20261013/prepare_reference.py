"""Record actual head metrics and immutable inputs without modifying references."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
IFMDD = Path('/work/gm64/m64000/IF-MDD')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths = [
        next((IFMDD / 'exp_librispeech100_zipa_ipa_monophone_film_topo').glob('s3_t2_star_star/*/train_log.txt')),
        next((IFMDD / 'exp_librispeech100_10ms_budget256_ctxctc_film_20ep').glob(
            's3_t2_star_star/*/content_head_only_from_ckpt030_s1t1content049_10ep/train_log.txt')),
        next((IFMDD / 'exp_librispeech100_10ms_monophone_ctc_probe_from_b256_content040').glob(
            's3_t2_star_star/*/ctc_lin_only_mono_from_content040_1gpu/train_log.txt')),
    ]
    runs = []
    for path in paths:
        entries = []
        for line in path.read_text().splitlines():
            match = re.search(r'valid content_PER:\s*([\deE.+-]+)', line)
            if match:
                entries.append((float(match.group(1)), line))
        best = min(entries)
        config = path.with_name('hyperparams.yaml')
        runs.append(dict(directory=str(path.parent), log_sha256=sha(path),
                         config_sha256=sha(config), best_recorded_content_PER=best[0],
                         original_log_line=best[1], metric='historical validation content PER, not TIMIT core-test'))
    sources = [IFMDD / 'utils/LossManager.py', IFMDD / 'utils/DataPrepIO.py',
               IFMDD / 'utils/subphonetic_topology.py',
               IFMDD / 'models/phn_mono_ssl_model_v3_refactored.py']
    payload = dict(
        requested_head='FiLM experiment content/CTC head only', runs=runs,
        criterion='speechbrain.nnet.losses.ctc_loss(blank_index=0, reduction=mean, zero_infinity=True)',
        decoder='speechbrain.decoders.ctc.ctc_greedy_decode(blank_id=0)',
        targets='existing MFA IPA105 manifest, explicitly retained by user; no label conversion',
        architecture='native Zapformer encoder + existing 10ms Fbank frontend + single 105-class nn.Linear',
        scope='content branch migration; no topology/fused FiLM losses or learned OT masses',
        joint_reference_limitation='The historical 3.17% run jointly trained the SSL encoder with other heads; a CTC-only Zapformer pilot does not reproduce its complete training setup.',
        encoder_auxiliary='disabled by default to isolate the IF-MDD CTC objective; optional explicit override',
        ctc_backend='SpeechBrain/PyTorch CTC on CPU via differentiable score copy to retain deterministic CUDA training; unchanged objective and reduction',
        source_hashes={str(path): sha(path) for path in sources},
        historical_files_modified=False,
    )
    (HERE / 'reference.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({run['directory']: run['best_recorded_content_PER'] for run in runs}, ensure_ascii=False))


if __name__ == '__main__':
    main()
