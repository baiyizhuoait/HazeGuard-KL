# 质检通告 005：收尾修复（1 号，2026-11-17）

数据链已经定稿，这一轮只动“跑不起来、看出来是旧的、口径没标”的地方，没有改任何实验结论。

## 1. 运行问题

| 项 | 现象 | 处理 |
|---|---|---|
| M1 产物路径 | `metrics_stage1.json` 的 `panel_path` / `eligibility.source` 写的是本机绝对路径，换台机器就失效 | 改成仓库相对 POSIX 路径（`data/haze_panel_clean.parquet`），`model_path` 一并统一；新增测试断言产物里没有盘符和反斜杠 |
| Streamlit 依赖 | `requirements.txt` 里 `streamlit` 是注释状态，`pip install -r` 之后 `app.py` 起不来 | 启用 `streamlit>=1.40`（altair 随它安装）；plotly 等研究依赖仍保持注释 |
| 区间回退 | adaptive 文件缺席时退回 calibrated 文件，`test.by_node[node]["PICP"]` 抛 `KeyError: 'PICP'` | 见下节 |

### 区间回退的根因

两份区间的嵌套层级不同：

- `metrics_stage5_interval_adaptive.json` → `test.overall.PICP`，`by_node[node].PICP`
- `metrics_stage5_interval_calibrated.json` → `test.calibrated.PICP`，`by_node[node].calibrated.PICP`

原代码按第一种写死，回退到第二种就取不到键。已抽出 `src/interval_view.py`，`pick_block`
逐层找真正带 `PICP` 的那一层，`by_node` 也两种都接受；`app.py` 与 `tools/verify_stage5.py`
共用它。`tests/test_interval_view.py` 4 项 + `tests/test_app_smoke.py` 2 项覆盖，
其中第二条把 adaptive 文件临时改名，让回退路径真的被执行。

## 2. 系统分析的新鲜度

`metrics_stage7.json` 是派生结果，读的是缓存概率、冻结阈值和接缝报告。上游重算之后它照样
能渲染，页面会把旧结论摆在新数字旁边。已做两件事：

1. 用当前缓存重算，旧版归档到 `docs/qc_fix/pre_refit_stage7/`；
2. 新增 `src/cache_freshness.py`，页面加载前逐个比对 `input_sha256`，任何一项不符就不渲染
   那三段（早预警、校准、误差解剖），改为提示重跑 `python src/07_system_analysis.py`。

重算后 190 个数值变化的主要来源是 `rows_by_split.val` 从 16,656 变成 16,647，以及
M5 早预警的分档计数（例如 `lead_counts.3` 225 → 224、`alert_precision` 0.8267 → 0.8309）。

## 3. 新旧实验的区分

主表加了一列 `result_version`（契约 11 → 12 列，spec v3.3）：

| 取值 | 模型 | 含义 |
|---|---|---|
| `qc_refit` | M1、M2、M3、M4、M5_fixed、M5_val_tuned、M7 | 在 QC 面板上重训或重算 |
| `pre_qc` | M6、M8 | M6 沿用旧模型旧预测；M8 是旧面板上的 v2 对照 |

M7 保留 `qc_refit`：它的阈值在 QC 验证集上重选、分数用当前概率重算，只是恰好落在同一个
操作点上，所以数值与旧值相同。这跟“沿用旧结果”不是一回事，标签上要区分开。

界面同步：`app.py` 的主表说明改成逐类解释，并删掉原来那句“every model here uses feature
table v1”——那句话既不准确，也和分层展示矛盾。

## 4. 4 号的 CV

`chronological_splits` 只按时间戳切折，训练行末尾 3 小时的标签落进验证窗口。已加
horizon purge，`tests/test_stage4_cv.py` 3 项（每折标签严格早于验证起点、净空确实删了行、
时间戳太少时抛错）。

固定原参数重算，结果另存 `models/metrics_stage4_cv_fixed.json`，旧搜索记录未动：

| 模型 | 修复前 CV | 修复后 CV | 差 |
|---|---:|---:|---:|
| M2 名义 RF | 0.6746 | 0.6747 | +0.0001 |
| M3 平衡 RF | 0.6746 | 0.6730 | −0.0016 |
| M4 Boosting | 0.6737 | 0.6694 | −0.0043 |

M4 受影响最大，与它的样本加权方式一致。验证集与测试集指标不受影响：模型训练用的是完整
训练段，这次只改了折内评估。

## 5. 验收

```
python -m pytest tests/ -q                 # 135 passed, 8 subtests
python src/01_verify_data.py --check       # 17/17
python tools/verify_stage5.py              # 3 项 PASS
python -m pytest tests/test_app_smoke.py   # 默认加载 + 区间回退各 1 项
```

`known_defects.json` 新增 KD-07（4 号 CV），KD-01~07 全部有修复记录或明确状态。

## 6. 报告同步（这轮之后才发现的）

stage7 重算改了 190 个数值，报告里有几处引用的是旧值，一并改掉：

| 文件 | 改动 |
|---|---|
| `reports/sec3_cost_and_explain.md` | 阈值选择用的验证集 16,656 → **16,647**（阈值确实是在新验证集上重选的） |
| `reports/sec4_system_and_demo.md` | 验证高危行数分母 16,656 → 16,647；早预警「发出预警」79.2% → **78.8%**、「预警确认」82.7% of 427 → **83.1% of 414**；区间 25.4/32.1 → **25.2/31.7**、88.0% → **88.1%**、KL_City 87.7/84.8 → **87.4/84.6**、固定半宽 83.4% → **83.3%**；误差解剖四档均值 +3.70/+1.92/−1.29/−16.10 → **+3.65/+1.87/−1.41/−16.29**、整体 MAE 8.07 → **8.05**，趋势表六个数同步 |
| `docs/spec_v3.md` | 可建模行 77,753/16,656 → **77,744/16,647**（两处），5 号对照的 val 行数同步 |

没有改的是 `reports/sec2_6_and_2_7.md` 第 9 行：「原始 M5 结果用了 77,753 / 16,656 / 16,647」——
这句描述的是换面板之前的那次实验，是历史事实；紧接着第 104 行已经说明 QC 表是 16,647。

其余章节核对后不需要改：sec2_3（2 号）与 sec2_4_2（3 号）的特征表没动；
sec2_6_models（4 号）没有引用 CV 具体值，正文只说明「交叉验证不是完全嵌套的」，
这句话在修完净空之后仍然成立。

`src/07_m8_v2_control.py` 会重写主表，它的列常量同步加了 `result_version`，
否则下次跑 M8 会把新列删掉。测试 `test_m8_writer_uses_the_same_contract` 守住这条。
