"""
One sanity test for estimate_pipeline's arithmetic, per the acceptance
criteria ("unit-tested with one sanity test").
Run: python -m tests.test_estimate  (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import assumptions
from agent.tools import estimate_pipeline


def test_all_digital_has_no_ocr_cost():
    result = estimate_pipeline(pdf_count=10, total_mb=1000, scanned_ratio=0.0)
    assert result["pages"]["scanned"] == 0
    assert result["extraction"]["api_path"]["cost_usd"] == (0.0, 0.0)
    assert result["extraction"]["gpu_path"]["cost_usd"] == (0.0, 0.0)
    expected_digital_pages = round(1000 / assumptions.MB_PER_DIGITAL_PAGE)
    assert result["pages"]["digital"] == expected_digital_pages


def test_all_scanned_has_zero_digital_pages_and_positive_ocr_cost():
    result = estimate_pipeline(pdf_count=5, total_mb=500, scanned_ratio=1.0)
    assert result["pages"]["digital"] == 0
    expected_scanned_pages = round(500 / assumptions.MB_PER_SCANNED_PAGE)
    assert result["pages"]["scanned"] == expected_scanned_pages
    low, high = result["extraction"]["api_path"]["cost_usd"]
    assert 0 < low <= high


def test_ranges_are_always_low_le_high():
    result = estimate_pipeline(pdf_count=20, total_mb=1500, scanned_ratio=0.6)
    for stage in ("download",):
        low, high = result[stage]["time_hr"]
        assert low <= high
    for path in ("api_path", "gpu_path"):
        low, high = result["extraction"][path]["time_hr"]
        assert low <= high
        low, high = result["extraction"][path]["cost_usd"]
        assert low <= high
    low, high = result["chunk_embed"]["cost_usd"]
    assert low <= high
    low, high = result["storage"]["postgres_gb"]
    assert low <= high
    for path in ("api_path", "gpu_path"):
        low, high = result["totals"][path]["cost_usd"]
        assert low <= high
        low, high = result["totals"][path]["time_hr"]
        assert low <= high


def test_more_mb_never_decreases_cost():
    small = estimate_pipeline(pdf_count=10, total_mb=500, scanned_ratio=0.5)
    big = estimate_pipeline(pdf_count=10, total_mb=1500, scanned_ratio=0.5)
    assert big["totals"]["api_path"]["cost_usd"][1] >= small["totals"]["api_path"]["cost_usd"][1]
    assert big["totals"]["gpu_path"]["cost_usd"][1] >= small["totals"]["gpu_path"]["cost_usd"][1]


if __name__ == "__main__":
    tests = [
        test_all_digital_has_no_ocr_cost,
        test_all_scanned_has_zero_digital_pages_and_positive_ocr_cost,
        test_ranges_are_always_low_le_high,
        test_more_mb_never_decreases_cost,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print("All sanity tests passed.")
