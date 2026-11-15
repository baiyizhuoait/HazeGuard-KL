# 质检通告 004：QC 修复已采纳，各模块从这里接

**2026-11-15 晚，`main` = `e8304bf`。** 插值规则与切分冲刷两条修复正式落地，
正式面板换成 966,406 行。下面按人写清楚要做什么，不用再回头翻通告 003 的长文。

## 0. 变了什么（一句话版）

| | 旧 | 新 |
|---|---|---|
| 面板行数 | 966,677 | **966,406** |
| 面板 SHA256 | `97b5db12…` | **`5cd7c1d3…`** |
| 受体 train / val / test | 78,102 / 16,737 / 16,728 | **78,093 / 16,728 / 16,728** |
| 可建模受体（02 口径） | 77,753 / 16,656 / 16,647 | **77,744 / 16,647 / 16,647** |
| 插值单元格 / 缺失行 | 656 / 277 | **299 / 489** |
| M1 交叉验证 | 0.6315（泄漏折） | **0.6277**（std 0.0107） |
| M1 六项测试指标 | 0.6658 / 0.7675 / 0.9047 … | **不变**（16,647 键逐行 0 差异） |

测试段受体一行没少；`target_level_3h` 与 `target_pm25_3h` 在共同键上**没有任何一行改变**。
变的是：271 行被剔除（156 边界冲刷 / 102 无目标 / 13 无锚点）、119 个 o3/co 单元格
由误补值改为 NaN、气候态重算导致 9,594 行 anomaly 变化（最大 1.02 µg/m³）。

## 1. 给 2 号

1. **把你的候选 02 提升为正式产物**：`data/feat_spatiotemporal.parquet` 现在必须对应
   966,406 行的新面板。你的候选（`data/candidate/stage2_qcfix/`）已经是在这块面板上建的，
   把它挪成正式目录即可，不用重算。
2. **改你自己的文档数字**：`docs/technical_spec_26grid.md`、`docs/repository_audit_stage2.md`
   里的 966,677、97b5db12、676,884/145,054/144,739 都过期了；契约以 `src/config.py`
   的 `PANEL_CONTRACT` 为准（77,744 / 16,647 / 16,647）。
3. **重登记指纹**：`data/feature_manifest.json` 要跟着新特征表更新，否则 `verify_data`
   的「特征表指纹」一项会红。
4. 你审计里那 4 个远场列（测试段最多 1,896 行、最大 65.18 µg/m³）的变化，请在交接里
   保留一句，3 号要拿它解释特征选择为什么必须重建。

## 2. 给 3 号

**先等 2 号把正式 02 推上来**（看 `data/feat_spatiotemporal.parquet` 的行数变成 966,406），
再开工。开工后注意三件事：

1. **旧 Top-25 不能沿用**。候选面板上共同合格受体的白名单列逐值一致，但选样集合变了、
   测试段远场四列变了，必须重新拟合、重新选。
2. **测试集不参与选特征**，沿用你原来的时间验证方案。
3. 产物写独立目录，先别覆盖 `data/features_final.parquet`，等 4 号确认接口再接。

预期影响：你的 12 段消融里「空间加权」和「邻域 50 列」两段最可能动，因为源区缺失变了
（`Palangkaraya` 少 115 行，`Jambi`/`Palembang` 各多二十来行 NaN）。

## 3. 给 4 / 5 / 6 号

等 02 正式表 + 3 号的特征表都到位再重训，顺序不能乱。三条纪律：

1. **不许按测试成绩改方案**。这次是数据修复导致的重跑，不是调参窗口；超参、阈值搜索
   的候选集合、区间校准方法都保持原样。
2. 报告里要写清「修复后重评」，不能再说「测试集从未查看」。
3. 主表口径仍是可建模行 16,647；如果某一行数字变了，写清楚是数据变化还是口径变化。

预期：M1 六项指标没变，所以 M2–M7 大概率只在小数点后第 3–4 位动；真正会动的是
用了远场特征的模型（若有）。

## 4. 给 7 号（若回来）

- 看板读的 `models/baseline_dt.joblib` 已经被新面板的模型替换（叶 384、深度 9，指标不变），
  不需要改代码。
- 报告里的 CV 数改成 **0.6277**；测试六项指标不动。
- `reports/HazeGuard_KL_Demonstration_Report.md` 仍然是空的，闸门因此判 07 `PARTIAL`。
  A7 还是唯一让总判定 FAIL 的项。

## 5. 验收口径（谁都可以自己跑）

```bash
python src/01_verify_data.py --check        # 17/17
python -m pytest tests/test_qc_rules.py     # 7 passed
python tools/check_prose.py reports/        # 0 硬性问题
python tools/check_report_numbers.py        # 只剩差值待复核
```

M1 的训练输入指纹在 `models/metrics_stage1.json`：
`panel_sha256 = 5cd7c1d3…`、`fit_input_sha256 = 1714a634…`、`fit_rows = 78,088`；
主表口径（可建模行）就是保存下来的 `baseline_dt.joblib`：`fit_rows = 77,744`，
模型文件里带同一份气候态（`preprocess.sha256`），重新加载能复现 `metrics_test_eligible`。
全受体行模型另存 `baseline_dt_allrows.joblib`，两模型在同一批测试键上的逐行差异
已由脚本计算并写入 `metrics_stage1.json → eligibility.vs_all_rows_diff_rows`。

## 6. 证据留在哪

| 位置 | 内容 |
|---|---|
| `docs/quality_notice_003.md` | 修复过程、271 行拆分、逐行预测比对 |
| `docs/qc_fix/evidence.json` | 对账与预测比对的机器可读版 |
| `docs/qc_fix/rows_dropped_271.csv` | 271 行逐行原因 |
| `docs/qc_fix/report_impact.md` | sec1 改了哪 15 处 |
| `docs/project_log.md` | 「11-15 晚」一节 |
