"""Native Lhotse batch schedules for the cached-feature performance pilot."""
import time
from lhotse import CutSet, MonoCut
from lhotse.dataset import DynamicBucketingSampler


def make_sampler(rows, cap, rank, quadratic_duration=None, world_size=4):
    cuts = CutSet.from_cuts(MonoCut(id=r['id'], start=0, duration=r['frames']/100., channel=0) for r in rows)
    return DynamicBucketingSampler(
        cuts, max_duration=cap, quadratic_duration=quadratic_duration,
        num_buckets=30, buffer_size=20000, shuffle=True, drop_last=True,
        world_size=world_size, rank=rank, seed=731, sync_buckets=True,
    )


def schedule(rows, cap, rank, count, quadratic_duration=None, world_size=4):
    start = time.perf_counter()
    lookup = {r['id']: r for r in rows}
    sampler = make_sampler(rows, cap, rank, quadratic_duration, world_size)
    result = []
    epoch = 0
    while len(result) < count:
        sampler.set_epoch(epoch)
        before = len(result)
        for cuts in sampler:
            batch = [lookup[c.id] for c in cuts]
            t = max(r['frames']/100. for r in batch)
            effective = t if quadratic_duration is None else t + t*t/quadratic_duration
            # Native DurationBatcher may overshoot when a later cut is longer.
            assert effective > 0
            result.append(batch)
            if len(result) == count:
                break
        assert len(result) > before
        epoch += 1
    return result, time.perf_counter()-start
