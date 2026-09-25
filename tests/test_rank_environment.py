from datapath_doctor.collectors.dataloader_profiler import DataLoaderProfiler


def test_rank_zero_from_environment_is_preserved(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("SLURM_PROCID", "7")
    monkeypatch.setenv("SLURM_LOCALID", "7")

    profiler = DataLoaderProfiler([b"batch"], monitor_workers=False)
    assert profiler.rank == 0
    assert profiler.local_rank == 0
    assert profiler.world_size == 8
