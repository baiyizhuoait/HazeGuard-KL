"""主表契约：列名、结果版本与特征表归属。

`result_version` 这一列是 11-17 加的：面板换过之后，M1–M5 与 M7 是在新面板上重训/重算的，
M6 沿用了旧模型的预测，M8 是旧面板上的 v2 对照。三种来源混在同一张表里而不标出来，
读者会把它们当成同一次实验的结果，所以列级契约要由测试守住。
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "models" / "all_models_benchmark.csv"

EXPECTED_COLUMNS = [
    "model",
    "feature_set",
    "result_version",
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


def test_result_versions_are_declared_per_model(benchmark):
    """M6 与 M8 必须是 pre_qc，其余是 qc_refit；不许出现空值或第三种类别。"""
    assert set(benchmark["result_version"]) == {"qc_refit", "pre_qc"}
    pre_qc = set(benchmark.loc[benchmark["result_version"] == "pre_qc", "model"])
    assert pre_qc == {"M6_Ordinal_RF", "M8"}
    qc = set(benchmark.loc[benchmark["result_version"] == "qc_refit", "model"])
    assert {"M1", "M2_Nominal_RF", "M3_Balanced_RF", "M4_Boosting", "M7"} <= qc


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
