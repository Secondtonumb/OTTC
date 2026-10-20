"""Exact verified historical frame-target function, isolated from model imports."""
import torch

def targets(rows, lengths, device):
    out = torch.full((len(rows), int(lengths.max())), -100, device=device, dtype=torch.long)
    for b, row in enumerate(rows):
        n = int(lengths[b])
        duration = float(torch.tensor(row['duration'], dtype=torch.float32))
        for label, start, end in row['intervals']:
            assert 3 <= label <= 104
            start = float(torch.tensor(start, dtype=torch.float32))
            end = float(torch.tensor(end, dtype=torch.float32))
            if end <= start:
                continue
            first = int(torch.floor(torch.tensor(start / duration * n, dtype=torch.float32)))
            last = int(torch.ceil(torch.tensor(end / duration * n, dtype=torch.float32)))
            first = max(0, min(n - 1, first))
            last = max(first + 1, min(n, last))
            out[b, first:last] = label
    return out
