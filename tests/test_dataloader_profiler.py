import time

import pytest

from datapath_doctor.collectors.dataloader_profiler import DataLoaderProfiler


def _slow_loader(n, delay):
    for i in range(n):
        time.sleep(delay)
        yield i


def test_profiler_records_data_wait_and_compute_time():
    profiler = DataLoaderProfiler(_slow_loader(5, 0.01), num_workers=2, prefetch_factor=2, monitor_workers=False)
    for _ in profiler:
        time.sleep(0.02)  # simulate compute
        sample = profiler.record_compute_done(queue_depth=3)
        assert sample.data_wait_s >= 0.0
        assert sample.compute_time_s >= 0.0
        assert sample.step_time_s == pytest.approx(sample.data_wait_s + sample.compute_time_s)

    assert len(profiler.samples) == 5
    assert [s.step for s in profiler.samples] == [0, 1, 2, 3, 4]
    assert profiler.samples[0].queue_capacity == 4


def test_record_compute_done_without_next_raises():
    profiler = DataLoaderProfiler(_slow_loader(1, 0.0), monitor_workers=False)
    iter(profiler)
    with pytest.raises(RuntimeError):
        profiler.record_compute_done()
