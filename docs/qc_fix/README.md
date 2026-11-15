# 交接包：插值规则 + 切分冲刷（候选面板）

给 2 号的读取说明。**这是候选数据，不是 main 的数据。** main 上的 `data/haze_panel_clean.parquet`
没有改动，`models/*.json` 也没有改动。

## 1. 文件

| 文件 | 内容 | 大小 |
|---|---|---|
| `data/candidate/haze_panel_qcfix.parquet` | 修复后的整块面板，966,406 × 35，列名与旧面板完全一致 | 24 MB |
| `docs/qc_fix/drops.parquet` | 新流程剔除的 586 行，带 `reason`（split_purge / no_target / no_anchor）与所属 `split` | 6 KB |
| `docs/qc_fix/rows_dropped_271.csv` | 相对旧面板**净减少**的 271 行，逐行给原因 | 13 KB |
| `docs/qc_fix/evidence.json` | 行数对账、271 行拆分、M1 逐行预测比对、训练输入指纹 | 2.5 KB |
| `docs/qc_fix/metrics_stage1_candidate.json` | 候选面板上的 M1 指标（含 `fit_input_sha256`） | 5.7 KB |
| `docs/qc_fix/cleaning_report_candidate.json` | 候选面板的清洗计数（含新字段 `rows_dropped_split_purge`、`gap_rule`） | 11 KB |
| `docs/qc_fix/gate0_report_candidate.json` | 候选面板的 Gate 0（四项全过） | 2 KB |
| `docs/qc_fix/fingerprints.json` | 上述文件与当前基线模型的 SHA256 | 1 KB |

## 2. 怎么读

```python
import pandas as pd
df = pd.read_parquet("data/candidate/haze_panel_qcfix.parquet")
df.head()
df.groupby(["role", "split"]).size()
```

列名、dtype、阈值、单位都没变，`split` 仍然只有 `train` / `val` / `test` 三个值
（`purged` 只在写盘前存在于内存，落盘前已经剔除）。

## 3. 校验

```python
import hashlib
h = hashlib.sha256(open("data/candidate/haze_panel_qcfix.parquet", "rb").read()).hexdigest()
assert h == "5cd7c1d3149147b8..."   # 完整值见 fingerprints.json
```

或者用 `docs/qc_fix/fingerprints.json` 逐项比对。

## 4. 重新生成

不需要原始 API 缓存也能核对：`docs/qc_fix/rows_dropped_271.csv` 给出全部被剔除的键，
`evidence.json` 给出每一项计数。若要完整重建（需要本机 `data/raw/` 缓存）：

```bash
python src/01_data_pipeline.py --out-dir <输出目录> --dump-drops
python src/01_baseline_dt.py --panel <输出目录>/data/haze_panel_clean.parquet \
       --out-dir <输出目录> --fixed-params
python tools/verify_qc_fix.py --new <输出目录>/data/haze_panel_clean.parquet \
       --drops <输出目录>/models/drops.parquet --out-dir docs/qc_fix
```

第二条命令耗时约 3 秒，第一条约 1 分钟（26 个节点从缓存读）。

## 5. 你应该先看的三件事

1. **源区缺失**：`Palangkaraya` 的 37 小时缺口不再被补前 3 个值，整段为 NaN，连带少 115 行；
   `Jambi` +22、`Palembang` +18 行新缺失。上风向特征（`upwind_*`）的 NaN 比例会变。
2. **受体只有 3 行受影响**：`Sepang` 2023-03-08 14:00–16:00 的 `o3` 由「误补值」变 NaN（train 段）。
3. **基线相减列重算**：切分冲刷掉了 156 行 train，气候态均值随之微调，9,594 行的
   `b_proto_pm25` / `pm2_5_anomaly` 有变化，最大 1.02 µg/m³。**这是测试特征变化的原因**，
   不要以为是同一份输入。

## 6. 契约差

| 项 | 旧 | 候选 |
|---|---|---|
| 总行数 | 966,677 | 966,406 |
| 受体 train / val / test | 78,102 / 16,737 / 16,728 | 78,093 / 16,728 / 16,728 |
| 插值单元格 | 656 | 299 |
| 仍缺失行 | 277 | 489 |
| 剔除合计 | 315（226 + 89） | 586（156 + 328 + 102） |

`steps 3–6` 的计数仍然先于剔除记录，所以 `01_verify_data.py` 重算会得到不同的插值行数，
这是口径问题，不是新错误。

## 7. 一句话

面板可以当普通面板用；需要改的是 `PANEL_CONTRACT` 的三段行数与 `verify_data` 的期望值，
这两处等 2 号确认后再动。
