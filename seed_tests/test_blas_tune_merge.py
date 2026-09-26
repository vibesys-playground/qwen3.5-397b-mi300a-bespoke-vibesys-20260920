"""CPU tests for `SEED_BLAS_TUNE_MERGE`'s table handling (blas_tune.py): the merge keeps the
fastest solution per shape across ranks and round-trips TunableOp's CSV format.

    <python-with-torch> -m pytest seed_tests/test_blas_tune_merge.py -q -o addopts=
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import blas_tune  # noqa: E402

VALIDATORS = "Validator,PT_VERSION,2.9.0\nValidator,ROCM_VERSION,7.0.0\n"
KEY = "GemmTunableOp_BFloat16_TN"


def write(path: Path, rows: list[tuple[str, str, float]]) -> Path:
    body = "".join(f"{KEY},{params},{sol},{ms}\n" for params, sol, ms in rows)
    path.write_text(VALIDATORS + body)
    return path


def test_merge_keeps_the_fastest_solution_per_shape(tmp_path: Path) -> None:
    a = write(
        tmp_path / "tunableop_gfx942_0.csv",
        [("tn_1_16", "Gemm_Rocblas_1", 0.0221), ("tn_2_16", "Gemm_Hipblaslt_7", 0.010)],
    )
    b = write(
        tmp_path / "tunableop_gfx942_1.csv",
        [("tn_1_16", "Gemm_Hipblaslt_3", 0.0097), ("tn_3_16", "Gemm_Rocblas_9", 0.02)],
    )
    va, ta = blas_tune.read_table(a)
    _, tb = blas_tune.read_table(b)
    merged = blas_tune.merge_tables([ta, tb])
    assert merged == {
        (KEY, "tn_1_16"): ("Gemm_Hipblaslt_3", 0.0097),
        (KEY, "tn_2_16"): ("Gemm_Hipblaslt_7", 0.010),
        (KEY, "tn_3_16"): ("Gemm_Rocblas_9", 0.02),
    }
    out = tmp_path / "merged.csv"
    blas_tune.write_table(out, va, merged)
    assert blas_tune.read_table(out) == (va, merged)
    assert out.read_text().startswith(VALIDATORS)


def test_rank_files_skip_staged_and_other_tables(tmp_path: Path) -> None:
    for name in (
        "tunableop_gfx942_0.csv",
        "tunableop_gfx942_3.csv",
        "tunableop_gfx942_0.tune.csv",
        "tunableop_chunked_gfx942_0.csv",
    ):
        (tmp_path / name).write_text(VALIDATORS)
    got = [p.name for p in blas_tune._rank_files(tmp_path, "gfx942")]
    assert got == ["tunableop_gfx942_0.csv", "tunableop_gfx942_3.csv"]


def test_missing_file_is_an_empty_table(tmp_path: Path) -> None:
    assert blas_tune.read_table(tmp_path / "nope.csv") == ([], {})


def test_vocab_shards_take_the_full_heads_solution() -> None:
    from types import SimpleNamespace

    import torch

    model = SimpleNamespace(lm_head=torch.empty(8, 4), tp=SimpleNamespace(world=4))
    table = {
        (KEY, "tn_32_16_4_ld_4_4_32"): ("Gemm_Hipblaslt_5", 0.5),
        (KEY, "tn_8_16_4_ld_4_4_8"): ("Gemm_Rocblas_1", 0.15),
        (KEY, "tn_8_64_4_ld_4_4_8"): ("Gemm_Rocblas_2", 0.4),  # no full entry: kept
        (KEY, "tn_8_16_8_ld_8_8_4096"): ("Gemm_Rocblas_3", 0.1),  # another GEMM: kept
    }
    got = blas_tune.pin_vocab_shards(table, model)
    assert got[(KEY, "tn_8_16_4_ld_4_4_8")] == ("Gemm_Hipblaslt_5", 0.15)
    assert got[(KEY, "tn_8_64_4_ld_4_4_8")] == ("Gemm_Rocblas_2", 0.4)
    assert got[(KEY, "tn_8_16_8_ld_8_8_4096")] == ("Gemm_Rocblas_3", 0.1)
