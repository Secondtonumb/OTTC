"""Fail closed if a submitted recipe or shared dependency has changed."""
import json
from common import HERE, source_hashes

if __name__ == '__main__':
    expected=json.loads((HERE/'SOURCE_CONTRACT.json').read_text())
    actual=source_hashes()
    assert actual==expected, 'Source contract changed; create a new recipe instead of resuming.'
    print('SOURCE_CONTRACT_PASS',flush=True)
