"""主表契约：`models/all_models_benchmark.csv` 必须保持 11 列，且 M8 标 v2。

这条测试的由来：11-17 那次重跑里 `feature_set` 列被顺手删掉了，CSV 看起来仍然正常，
但看板的 ``benchmark[benchmark["feature_set"] == "v1"]`` 会把 v2 的 M8 也算进主表，
于是「主表全部是 v1」的说明和表内容自相矛盾。列级契约必须由测试守住，不能靠人看。
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "models" / "all_models_benchmark.csv"

EXPECTED_COLUMNS = [
    "model",
    "feature_set",
    "split",
    "seed",
    "Macro_F1",
    "Balanced_Accuracy",
    "High_Risk_Recall",
    "MAOE",
    "Severe_Jumping_Error_Rate",
    "Avg_Asymmetric_Cost",
    "n_samples",
]


@pytest.fixture(scope="module")
def benchmark() -> pd.DataFrame:
    if not BENCHMARK.exists():
        pytest.skip("尚未生成主表")
    return pd.read_csv(BENCHMARK)


def test_columns_match_contract(benchmark):
    assert list(benchmark.columns) == EXPECTED_COLUMNS


def test_feature_set_is_only_v1_except_m8(benchmark):
    assert set(benchmark["feature_set"]) <= {"v1", "v2"}
    m8 = benchmark[benchmark["model"] == "M8"]
    assert len(m8) == 1 and m8["feature_set"].iloc[0] == "v2"
    others = benchmark[benchmark["model"] != "M8"]
    assert (others["feature_set"] == "v1").all()


def test_v1_main_table_excludes_m8(benchmark):
    """看板主表的取数与说明必须一致：v1 子集里不能出现 M8。"""
    main_table = benchmark[benchmark["feature_set"] == "v1"]
    assert "M8" not in set(main_table["model"])
    assert {"M1", "M2_Nominal_RF", "M3_Balanced_RF", "M4_Boosting", "M7"} <= set(
        main_table["model"]
    )


def test_every_row_uses_the_eligible_test_size(benchmark):
    """所有模型都应在可建模测试集 16,647 行上评估，否则跨模型比较不成立。"""
    assert set(benchmark["n_samples"]) == {16647}


def test_no_missing_cells(benchmark):
    assert not benchmark.isna().any().any()
