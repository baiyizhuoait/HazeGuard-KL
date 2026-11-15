# 02 模块 QC 候选交接

本目录交给成员 3 核对。旧的正式 02 产物继续从仓库根目录的 `data/feat_spatiotemporal.parquet` 读取。成员 1 可据本轮对账决定是否登记候选的行数与数据契约。

- 候选输入来自成员 1 的固定提交 `ec1ff0aa8a3a7bd4cfe776ddee28cf2d68853919`，SHA-256 为 `5cd7c1d3149147b85259bcf5d48dfda5a88d8cff7a42b4030ed2cb129e4a5bc6`。[01 候选面板、差异清单与指纹](https://github.com/baiyizhuoait/HazeGuard-KL/tree/ec1ff0aa8a3a7bd4cfe776ddee28cf2d68853919/docs/qc_fix)
- [候选特征目录](data/feat_spatiotemporal.parquet/)含 966,406 行、98 列。用 `pd.read_parquet` 读取整个目录。分片行数、大小和 SHA-256 见[特征审计](models/metrics_stage2_features.json)。
- [逐键审计](models/metrics_stage2_qc_review.json)记录每段新增与删除、资格变化的实际受体键、标签差异、63 个白名单字段的变化行数、缺失状态和最大有限值差。键为 `station_id × timestamp`，时间按 `Asia/Kuala_Lumpur` 理解。
- 候选五张图在 [figures](figures/) 下，生成数字与哈希见 [EDA 统计](models/metrics_stage2_eda.json)。五张图相对正式版的哈希均有变化，受体测试类别分布和固定配对的峰值时刻一致。

## 按键对账

| 切分 | 全网删除（受体 / 走廊 / 源区） | 新增键 | 合格受体（正式 → 候选） | 共同合格受体的标签变化 |
|---|---:|---:|---:|---:|
| train | 78（9 / 30 / 39） | 0 | 77,753 → 77,744 | 0 |
| val | 78（9 / 30 / 39） | 0 | 16,656 → 16,647 | 0 |
| test | 115（0 / 0 / 115） | 0 | 16,647 → 16,647 | 0 |

- 删除清单的 271 个键与实际差集一致，其中边界冲刷 156 个、缺目标 102 个、缺锚点 13 个。分段删除的 18 条受体边界行原已不合格。它们被删后，严格同段 `t+3h` 查找使此前相邻的训练、验证受体各 9 条失去资格。共同键的标签值、现存受体的时间滞后和滚动特征保持一致。保持当前目标检查，登记新资格数。
- Sepang 当地时间 2023-03-08 14:00–16:00 的 O3 变空，原 `qc_interpolated=1` 变为 `is_missing=1`。17:00–18:00 原已缺失。这五小时均被现有行级清洁检查排除，检查从当日 11:00 到次日 18:00 的共同键，资格、标签及 O3 以外的入模特征无变化。
- 非边界删除全部位于 Palangkaraya 的测试段，其中 33 个旧小时原可用。Jambi、Palembang 等源区改动涉及 O3/CO 和 QC 标记，所有保留键的 `audit_source_clean` 一致。三个受体的近场浓度、梯度、权重和计数逐值一致。源区删除影响 Borneo 本地计算及跨区域历史背景。表中的远场变化只出现在共同合格的测试受体，四列均没有新增缺失。
- 气候态 `b_proto_pm25`、`b_proto_temp`、`b_proto_n` 和两列异常值各有 9,594 个共同键发生变化，其中共同合格受体按 train/val/test 分别为 762/57/279。02 的白名单排除这些列。

| 共同合格测试受体上的远场字段 | 变化行数 | 最大绝对差（µg/m³） |
|---|---:|---:|
| `far_source_pm25_mean_12h` | 630 | 14.04851 |
| `far_source_pm25_mean_24h` | 1,149 | 9.63793 |
| `far_source_pm25_mean_48h` | 1,896 | 5.98029 |
| `far_source_pm25_lag_24h` | 99 | 65.17692 |

候选若获采纳，成员 3 应在独立目录重建粒化和特征选择。共同合格的训练和验证受体上，白名单列逐值一致，但选样集合变化，测试远场值也变化。旧特征选择结果不可直接移作候选结果。项目 v3 将邻域 50 列留作消融附录，候选交付沿用 02 的 63 列白名单。成员 1 登记契约后，再用候选结果衔接后续模块。

## 成员 3 先核对输入

在 `feat/02-eda` 的仓库根目录，使用已安装 `requirements-stage2.lock.txt` 的 Python 运行以下 PowerShell 命令。它只读候选数据和审计。

```powershell
@'
import json
from pathlib import Path
import pandas as pd

root = Path("data/candidate/stage2_qcfix")
audit = json.loads((root / "models/metrics_stage2_features.json").read_text(encoding="utf-8"))
features = audit["model_feature_columns"]
keys = ["station_id", "timestamp"]
labels = ["target_pm25_3h", "target_level_3h"]
table = pd.read_parquet(root / "data/feat_spatiotemporal.parquet",
                        columns=[*keys, "split", "role", "audit_model_eligible", *labels, *features])
assert len(table) == 966406 and len(features) == 63
assert not table.duplicated(keys).any()
receptors = table.loc[table.role.eq("receptor") & table.audit_model_eligible]
counts = receptors.groupby("split", observed=True).size().to_dict()
assert counts == {"train": 77744, "val": 16647, "test": 16647}, counts
print(counts)
'@ | python -
```

为粒化准备独立目录。在同一个 02 仓库根目录执行，产物写到仓库外。候选已通过这项检查，全表得到 10 个粒化字段，三个切分的合格受体均无粒化缺失。

```powershell
$scratch = Join-Path (Get-Location).Path '../hazeguard-03-qc-review'
$candidate = (Resolve-Path 'data/candidate/stage2_qcfix').Path
New-Item -ItemType Directory -Force $scratch | Out-Null
git fetch origin main
git archive --format=zip --output="$scratch/03-source.zip" 31f678d8615d1df7607a34e4b1767a784009cba8 src/03_granulation_features.py src/03_feature_selection.py src/stage2_storage.py src/config.py src/utils_metrics.py
Expand-Archive "$scratch/03-source.zip" "$scratch/code" -Force
python "$scratch/code/src/03_granulation_features.py" --repo "$candidate" --out "$scratch/outputs"
```

成员 3 在候选获采纳后，可继续执行下一条。它会重新拟合特征选择内部的评估器，请沿用 03 的环境、按时间验证方案比较结果，测试集不参与选特征。

```powershell
python "$scratch/code/src/03_feature_selection.py" --repo "$candidate" --out "$scratch/outputs"
```

## 复核 02 的对账

在 02 仓库根目录，先获取包含成员 1 固定提交的分支，再运行下列命令。对账过程验证上游指纹、分片、图、主键、原始字段及候选 Gate 0，最后把本次审计写入独立目录。Git 中的文本文件以 LF 储存，随附的 Windows 文本指纹按 CRLF 计算，审计中逐项注明。

```powershell
$scratch = Join-Path (Get-Location).Path '../hazeguard-02-qc-audit'
New-Item -ItemType Directory -Force $scratch | Out-Null
git fetch origin fix/qc-gap-rule-and-purged-cv
git archive --format=zip --output="$scratch/01-source.zip" ec1ff0aa8a3a7bd4cfe776ddee28cf2d68853919 data/candidate/haze_panel_qcfix.parquet docs/qc_fix
Expand-Archive "$scratch/01-source.zip" "$scratch/upstream" -Force
python tools/compare_stage2_qc.py --candidate data/candidate/stage2_qcfix --upstream "$scratch/upstream" --output "$scratch/comparison.json"
```

候选生成沿用 `2165c4cbe2965c49284f41214339ce187507c519` 的 02 源码。运行时在独立源码副本里使用 01 候选 Gate 0，两个生成器都显式传入候选面板，并指定独立的特征、图和 JSON 输出。所有版本的正式路径均保留。

```powershell
git archive --format=zip --output="$scratch/02-source.zip" 2165c4cbe2965c49284f41214339ce187507c519 src/02_spatiotemporal_features.py src/02_eda_analysis.py src/stage2_storage.py
Expand-Archive "$scratch/02-source.zip" "$scratch/runner" -Force
New-Item -ItemType Directory -Force "$scratch/runner/models" | Out-Null
Copy-Item "$scratch/upstream/docs/qc_fix/gate0_report_candidate.json" "$scratch/runner/models/gate0_report.json"
Set-Content -Encoding ascii "$scratch/runner/SOURCE_COMMIT.txt" '2165c4cbe2965c49284f41214339ce187507c519'
$panel = "$scratch/upstream/data/candidate/haze_panel_qcfix.parquet"
$rebuilt = "$scratch/rebuilt"
python "$scratch/runner/src/02_spatiotemporal_features.py" --input "$panel" --output "$rebuilt/data/feat_spatiotemporal.parquet" --audit "$rebuilt/models/metrics_stage2_features.json"
python "$scratch/runner/src/02_eda_analysis.py" --input "$panel" --figures "$rebuilt/figures" --metrics "$rebuilt/models/metrics_stage2_eda.json"
python tools/compare_stage2_qc.py --candidate "$rebuilt" --upstream "$scratch/upstream" --output "$scratch/rebuilt-comparison.json"
```
