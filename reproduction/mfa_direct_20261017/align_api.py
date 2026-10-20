"""Run the official MFA align command's core Python API on the four WAVs.

Inputs are already mono 16kHz PCM WAV. No resampling or Postgres is needed.
The acoustic model, dictionary, and inference implementation are unmodified.
"""
import importlib.metadata
import json
from pathlib import Path
import time

from montreal_forced_aligner.alignment import PretrainedAligner
from montreal_forced_aligner.config import GLOBAL_CONFIG

HERE = Path(__file__).resolve().parent


def main():
    version = importlib.metadata.version('Montreal-Forced-Aligner')
    (HERE / 'MFA_VERSION.txt').write_text(version + '\n')
    GLOBAL_CONFIG.current_profile.use_mp = False
    GLOBAL_CONFIG.current_profile.num_jobs = 1
    GLOBAL_CONFIG.current_profile.use_postgres = False
    started = time.monotonic()
    aligner = PretrainedAligner(corpus_directory=HERE / 'corpus',
        dictionary_path=HERE / 'models/english_us_arpa.dict',
        acoustic_model_path=HERE / 'models/english_us_arpa.zip',
        temporary_directory=HERE / 'api_work', num_jobs=1, use_mp=False,
        use_postgres=False, clean=False, debug=True)
    try:
        aligner.align()
        aligner.analyze_alignments()
        aligner.export_files(HERE / 'textgrids', output_format='long_textgrid', include_original_text=True)
        status = dict(tool_version=version, api='official PretrainedAligner.align/analyze_alignments/export_files',
                      elapsed_seconds=time.monotonic() - started, expected_utterances=4,
                      source_audio='mono 16kHz PCM WAV', resampling=False, use_postgres=False,
                      acoustic_model_unchanged=True, dictionary_unchanged=True)
        (HERE / 'ALIGN_API_COMPLETE.json').write_text(json.dumps(status, indent=2) + '\n')
        print(json.dumps(status), flush=True)
    finally:
        aligner.cleanup()


if __name__ == '__main__':
    main()
