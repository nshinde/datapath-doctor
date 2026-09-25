from datapath_doctor.collectors.dataloader_profiler import DataLoaderProfiler


def test_profiler_propagates_explicit_rank_context():
    profiler = DataLoaderProfiler(
        [b"batch"],
        monitor_workers=False,
        rank=3,
        local_rank=1,
        world_size=8,
        node_id="worker-b",
    )
    for _batch in profiler:
        sample = profiler.record_compute_done()

    assert sample.rank == 3
    assert sample.local_rank == 1
    assert sample.world_size == 8
    assert sample.node_id == "worker-b"
