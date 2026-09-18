import time

from datapath_doctor.collectors.proc_stats import DiskIOCollector, PSICollector

_FAKE_DISKSTATS_T0 = """   8       0 sda 1000 0 20000 500 200 0 4000 300 0 400 800
   8       1 sda1 900 0 18000 450 190 0 3800 280 0 380 730
"""

_FAKE_DISKSTATS_T1 = """   8       0 sda 1100 0 22000 600 220 0 4400 340 0 480 940
   8       1 sda1 990 0 19800 495 209 0 4180 308 0 418 803
"""


def test_disk_io_collector_computes_deltas(tmp_path):
    path = tmp_path / "diskstats"
    path.write_text(_FAKE_DISKSTATS_T0)

    collector = DiskIOCollector("sda", diskstats_path=str(path))
    assert collector.sample() is None  # first call just primes the baseline

    path.write_text(_FAKE_DISKSTATS_T1)
    # Force a nonzero dt by monkeypatching time isn't necessary here since
    # _read_counters() calls time.time() fresh each call; just ensure a real
    # elapsed interval so dt > 0.
    time.sleep(0.01)
    reading = collector.sample()

    assert reading is not None
    assert reading["disk_name"] == "sda"
    assert reading["disk_read_iops"] > 0
    assert reading["disk_read_bytes_per_s"] > 0
    assert reading["avg_read_size_bytes"] > 0


def test_disk_io_collector_unknown_device_returns_none(tmp_path):
    path = tmp_path / "diskstats"
    path.write_text(_FAKE_DISKSTATS_T0)
    collector = DiskIOCollector("nvme99n1", diskstats_path=str(path))
    assert collector.sample() is None


def test_disk_io_collector_missing_file_returns_none(tmp_path):
    collector = DiskIOCollector("sda", diskstats_path=str(tmp_path / "does-not-exist"))
    assert collector.sample() is None


_FAKE_PSI_IO = """some avg10=5.23 avg60=3.10 avg300=1.05 total=123456
full avg10=1.11 avg60=0.90 avg300=0.40 total=54321
"""


def test_psi_collector_parses_avg10(tmp_path):
    path = tmp_path / "io"
    path.write_text(_FAKE_PSI_IO)
    collector = PSICollector(path=str(path))
    reading = collector.sample()
    assert reading == {"psi_io_some_avg10": 5.23, "psi_io_full_avg10": 1.11}


def test_psi_collector_missing_file_returns_none(tmp_path):
    collector = PSICollector(path=str(tmp_path / "does-not-exist"))
    assert collector.sample() is None
