# 05 模块交接：回归分级、有序随机森林与树间区间

**当前交付以第 14 节为准：M5 已在 QC 表重评，自适应测试覆盖 88.09%、平均宽度 31.74 µg/m³。前面各节保留分日期的历史记录。**

本模块按 `docs/spec_v3.md` 第 4 节实施。数值来源为
`models/metrics_stage5.json`；学习说明及结果表见 `reports/sec2_6_and_2_7.md`。

## 1. 上游与运行

- 本地基于 main `7dfe3e825d7aef851db60680eb9eb0a636b03e46`。
- 输入取自 03 分支 `50b6e63dfc936945e36999c797e85bf7f5ba4696`。
- 特征表 SHA256：`64285f40e81c72ab4872d282d52be68401078e8c33dec50134d93e7877d2f55b`。
- 仅使用受体节点 KL_City、Port_Klang、Sepang；25 列与上游顺序相同。
- 行数（QC 面板，2026-11-17）：train 77,744 / val 16,647 / test 16,647；换面板前的旧值为 77,753 / 16,656，旧产物归档在 `docs/qc_fix/pre_refit_stage5/`。
- 未修改其他成员代码、配置、指标或数据；本地从 03/04 分支取出的文件只是依赖与对照。
- 公共依赖沿用 `requirements.txt`，不增加 quantile-forest、SHAP 或其他算法库。

在已经整合 01～03 产物的仓库根目录运行：

```sh
python src/05_ordinal_rf.py --validate-only
python -m unittest discover -s tests -p test_stage5.py -v
python src/05_ordinal_rf.py --n-jobs 8
python tools/verify_stage5.py
```

`python src/05_reg_to_cls.py` 是同一完整流程的另一入口，**不必接着再运行一次**。
已有 `metrics_stage5.json` 或 `stage5_selection.json` 时拒绝覆盖重跑，避免反复查看 test。
核对结果用 `verify_stage5.py`，它读取已有预测，不重新预测 test。
代码和 JSON 的摘要检查兼容 Git 检出时的 LF/CRLF 换行转换，二进制产物仍要求逐字节一致。
若确需独立复现实验，应使用另一份干净副本，并将其作为重复实验记录，而不是试着挑更好 test。

## 2. 实现

### M5 回归转分级

`RandomForestRegressor` 学习 `target_pm25_3h`。
固定分档为 15 / 35 / 55 µg/m³，边界值归到较高档。
另提供验证集阈值校准对照：每个切点相对固定值偏移 {-4,-2,0,2,4}，共 125 组，
以 val Macro-F1 最大为主规则；相同分数优先 MAOE 小、偏移小的组合。
**不改真实标签的定义**，校准只改预测的切点。
切点由种子 42 的 val 决定后，在种子 43/44 和 test 上固定使用。

### M6 有序 RF

`src/ordinal_forest.py` 定义可序列化的 sklearn 风格分类器。
三个 RF 分别学习 Y>0、Y>1、Y>2，使用 `balanced_subsample`。
先记录原始概率的单调违反数，再按规范向下累积取最小值：

```text
q0* = q0
q1* = min(q0, q1)
q2* = min(q0, q1, q2)
p = [1-q0*, q0*-q1*, q1*-q2*, q2*]
```

四级概率归一化后，`argmax` 为默认等级；概率相同时取较低编号。
训练折只有一个二分类类别时也能输出正确的常数概率。
该修正保证概率有效，不保证提升所有指标，也不是概率校准。
相对名义 RF，结果包含累积分解和二分类类权重变化的共同影响，不能全归因于次序。

### 搜索与时间验证

3 组 RF 结构，包含 04 选出的结构，并分别降低深度、减小叶节点样本数。
每组运行 train 内 5 折 TimeSeriesSplit，按唯一 timestamp 分组，三个节点同刻一起划分。
按真实时间清除 `train timestamp + 3h >= fold validation start` 的训练行。
候选最终以 val Macro-F1、较低 MAOE、CV Macro-F1、较浅树依次选定。
固定超参数后，种子 42/43/44 只在 train 拟合，在 val 报均值及样本标准差。
最终保存种子 42。所有选择冻结并落盘后才产生一批 test 预测。

局限：上游 Top-25 用全部 train 标签筛选，内部 CV 不构成完整的嵌套验证。
val/test 未参与特征选择。本模块没有重做特征选择，保证与前面模型使用相同输入。

### 浓度区间

用 **M5 回归森林** 各树的浓度点预测取 5 / 50 / 95 百分位。
报告 PICP（真实浓度落在上下界之间的比例）、MPIW（平均带宽，µg/m³）。
这反映树之间的预测分散程度，**不是经过校准的 90% 预测区间，也不是分位数随机森林**。
示例图固定为 Sepang 最早 168 个测试小时，横轴为 t+3h，缺失小时留空，不选最好看的事件。

## 3. 给 6 号的概率接口

`models/ordinal_probs_val_test.npz` 可 `allow_pickle=False` 安全读取：

| 键 | 形状 | 说明 |
|---|---|---|
| `proba_val`, `proba_test` | (n,4) | 修正并归一化后，列顺序 L0/L1/L2/L3 |
| `y_val`, `y_test` | (n,) | 真实等级，0～3 |
| `ts_val`, `ts_test` | (n,) | 无时区的吉隆坡本地时间 t；不是目标时刻 t+3h |
| `station_id_val`, `station_id_test` | (n,) | 同刻多个节点，联合 timestamp 才能唯一定位 |

val n=16,647；test n=16,647。行序均为 timestamp、station_id。
**采用 spec_v3 的 `proba_val`，不是旧文档的 `val_probs`。**
高危概率为 `proba[:,2] + proba[:,3]`。阈值只能用 val 调整，test 用于冻结方案的最终报告。

```python
import numpy as np
z = np.load("models/ordinal_probs_val_test.npz", allow_pickle=False)
p = z["proba_val"]
y = z["y_val"]
high_risk_score = p[:, 2:].sum(axis=1)
```

## 4. 给 7 号的模型接口

先将仓库 `src` 加入模块查找路径，确保自定义分类器可被加载：

```python
import sys
from pathlib import Path
import joblib
import numpy as np
root = Path.cwd()  # 在仓库根目录运行
sys.path.insert(0, str(root / "src"))
ordinal = joblib.load(root / "models/ordinal_rf.joblib")
reg = joblib.load(root / "models/rf_regressor.joblib")
X = rows[ordinal.selected_features_].to_numpy(dtype=np.float32)
probability = ordinal.predict_proba(X)
level = probability.argmax(axis=1)
concentration = reg.predict(X)
fixed_level = np.digitize(concentration, reg.level_thresholds_fixed_)
tuned_level = np.digitize(concentration, reg.level_thresholds_val_tuned_)
tree_values = np.stack([t.predict(X) for t in reg.estimators_], axis=1)
band = np.quantile(tree_values, [0.05, 0.5, 0.95], axis=1).T
```

输入必须已有 25 个有限特征，不能直接传 API 原始字段。
模型是历史 CAMS 模式产品上的教学实验，区间和目标均不是地面监测真值。

## 5. 产物

- 代码：两份数字前缀入口 + `src/ordinal_forest.py`。
- 模型：`rf_regressor.joblib`、`ordinal_rf.joblib`，均要求 <20 MiB。
- 下游接口：`ordinal_probs_val_test.npz`。
- 证据：`metrics_stage5.json`、`stage5_selection.json`、`stage5_predictions.npz`。
- 图：`fig_ordinal_error_distribution.png`、`fig_interval_band.png`，300 dpi。
- 检查：`tests/test_stage5.py`、`tools/verify_stage5.py`。
- 本交接及 `reports/sec2_6_and_2_7.md`（数据、公式与自主撰写提纲）。

`stage5_predictions.npz` 额外保存 M5 预测、区间、M6 修正前累积概率和三级对照预测，
便于复核数字和重画图，无须重新预测 test。
模型、输入、代码及概率文件的 SHA256 均登记在 metrics JSON 中。

## 6. 规范中需要组长确认的三处文字

1. §4.2 的浓度区间把模型写成 M6；本实现使用能输出浓度的 M5，未改共享规范。
2. §4.4 的 M3 val 0.8744/0.2822 已满足 0.85/0.29 门槛，不能宣称本阶段首次满足。
3. §4.5 只有概率矩阵是 (n,4)，真实标签、时间戳都是一维。

这些处理均写入 `metrics_stage5.json -> spec_clarifications`，没有静默改契约。

## 7. 已完成结果速查

val 三种子均值：M5 校准切点的高危召回 0.8600、MAOE 0.2644，满足两道数值门槛。
M6 对应 0.8439 / 0.3028，未满足。

test 固定种子 42：

| 方案 | Macro-F1 | 高危召回 | MAOE | 跨两级误判 | 平均代价 |
|---|---:|---:|---:|---:|---:|
| M5 固定切点 | 0.7046 | 0.8419 | 0.2750 | 0.0100 | 0.7180 |
| M5 验证集校准切点 | 0.7104 | 0.8789 | 0.2854 | 0.0091 | 0.7046 |
| M6 有序 RF（pre_qc，未重训） | 0.6839 | 0.8642 | 0.3169 | 0.0239 | 0.6616 |

本配置下，M5 校准方案相对 M6 的 F1、高危召回及跨级误判更好；M6 平均代价更低。
M6 的 Level 3 逐类召回为 0.8777，M5 为 0.7094；极端类别与总体序差存在取舍。
不据此声称所有有序分类方法都无用，也不将测试集结果用于继续改参数。

M6 单调违反率：val 3.48%、test 5.30%；修正后均为 0。
M5 树间区间 test PICP=0.7183、MPIW=20.7877 µg/m³，不支持“90% 真实浓度覆盖”的说法。

检查：4 项合成数据测试通过，基础数据 16/16 通过；保存结果复核通过，两张图已目视检查。
`check_prose.py` 没有硬性问题，但对中文数据笔记给了句长提示；它不验证原创性或课程许可。

模型文件各自小于 20 MiB，但两模型合计约 23 MB。完整交付压缩包供组内传递，
不等同于能直接上传 Moodle 的最终课程包；课程包可使用不含 joblib 的轻量包，模型按组长规范另存。

## 8. 可读性修改

删掉 `model_for`、`predict_for` 和 `regressor` 三层转接，训练处直接使用 sklearn 模型。
统计均值和标准差改成普通循环；四级概率分别写成 p0、p1、p2、p3；必要注释用简短中文解释。
完整训练入口的 `main()` 按读取、选参、验证、保存后测试、记录结果五步阅读。
部分重复的文字说明从训练入口移除，仍可在本交接查到。

未更改特征、候选参数、种子、时间折、类别权重、选参规则、切点或指标定义。
对照旧代码：合成数据上的训练、CV 和选参结果一致；全部 16,656 条验证预测等级一致，
概率最大绝对差约 5.6e-16（浮点累加误差）；已保存测试预测的后处理结果逐值一致。
没有重训真实模型或再次预测测试集。两个 joblib、概率/预测 NPZ、两张图和冻结记录的文件摘要均未改变。

`metrics_stage5.json` 保留原始训练时的 `code_sha256`，新增 `refactor` 字段登记当前代码和核对结果。
原版交付包保留；本次交付包文件名带 `_readable`。

## 9. 2026-11-13 区间校准补充：给 7 号

README 的“90% 预测区间”与原来的树间分散带存在口径差异。
原方法取单棵树预测的 5/50/95 百分位，没有用实际浓度误差校准覆盖率。
本补充只处理浓度区间；模型、分档阈值、原始分类结果及原有审计文件保留。

### 方法与实际结果

按反馈采用一个预先固定的简单方案，不比较多种方案后择优：

1. 取 M5 每条验证预测的树中位数 `m`，算绝对残差 `abs(y-m)`。
2. 对 n=16,656 个残差，取第 `ceil((n+1)*0.90)=14,992` 小的数。
3. 冻结半宽 `w=13.820877192982458 µg/m³`。
4. 新区间为 `[max(0,m-w), m+w]`。PM2.5 非负，截断负下界不损失有效目标的覆盖。
5. 冻结后直接应用于已有测试预测，报告覆盖率与宽度，不用测试目标反调 w。

| 切分 | 原树间带 PICP | 校准后 PICP | 原平均宽度 | 校准后平均宽度 |
|---|---:|---:|---:|---:|
| val | 71.68% | 90.01% | 16.29 | 26.87 |
| test | 71.82% | 83.29% | 20.51 | 27.21 |

宽度单位均为 µg/m³。验证集是计算宽度的数据，90.01% 属于样本内校准检查，不能当作泛化成绩。
新区间围绕中位数重新构造，并非逐条在原带两侧加同样的补偿量；不保证包含每条原树间带。
对每个预测，中位数不变、未截断的总宽度固定；靠近零时会缩短。

测试集分节点覆盖：KL_City 75.62%、Port_Klang 93.06%、Sepang 81.56%。
这是全节点混合校准，不能声称每个节点都达到 90%。

### 产物与接入

- `src/05_interval_calibration.py`：计算、冻结、应用校准；不调用 fit 或模型 predict。
- `models/interval_calibration.json`：冻结的半宽、名义覆盖率、输入与代码摘要。
- `models/metrics_stage5_interval_calibrated.json`：总体和分节点的前后对比。
- `models/stage5_intervals_calibrated.npz`：`interval_val/test` 为 (n,3)，列依次为下界、中位数、上界。
  `ts_val/test` 和 `station_id_val/test` 是行标识；时间仍为特征时刻 t，展示目标时刻要加 3 小时。
- `tests/test_stage5_interval.py`：有限样本分位点、冻结宽度推理、非负截断和异常输入检查。

离线页面可直接读新区间 NPZ。对新输入，先在 sys.path 中加入 src，再按原有特征顺序准备 X：

```python
import importlib
import json
from pathlib import Path
import numpy as np

R = importlib.import_module("05_reg_to_cls")
I = importlib.import_module("05_interval_calibration")
cal = json.loads(Path("models/interval_calibration.json").read_text(encoding="utf-8"))
median = np.median(R.tree_predictions(rf_regressor, X), axis=1)
band = I.apply_calibration(median, cal)  # 下界 / 中位数 / 上界
```

这里必须用 M5 回归森林的树中位数，不能把 M6 等级概率或 M5 的均值当作校准中心。
更换模型或输入数据后需要重新校准，不能复用旧半宽。
原 `fig_interval_band.png` 仍是明确标注的未经校准树间带；展示新区间时读取新 NPZ 绘制，不能给原图换标签冒充。

### 报告与页面必须保留的限制

建议页面标签：“90% 目标覆盖率区间（验证集校准；历史测试覆盖 83.41%）”。
7 号负责的总 README 应同步这个事实；本次未改共享 README 或 app.py。

不能写“已保证真实 90%”。测试期覆盖仍不足，且区间更宽，需同时报告 PICP 与 MPIW。
时间序列存在时间和空间相关性，不能直接套用独立可交换数据的严格覆盖保证。
原验证集也用于选模型，因此这是选参后的经验残差校准，不具备全新独立校准集的条件。
原测试成绩此前已知，本补充是事后扩展，不应包装成一次全新的盲测；没有拿测试覆盖率选宽度。
如要进一步研究动态校准，应预先规定更新规则，等 t+3h 目标真正可用后才更新，并使用新时期评估。
数据目标仍是 CAMS 模型产品，不是地面监测真值。

方法参考：[MAPIE 绝对残差分数](https://mapie.readthedocs.io/en/stable/content/conformal-prediction/conformity-scores/)，
[可交换性与时间/分布变化限制](https://mapie.readthedocs.io/en/stable/content/exchangeability-testing/)。

## 10. 后续改善：按树间分散程度自适应校准宽度

本节是固定半宽校准的改进版本。原始版本与第 9 节结果保留用于比较。

### 为什么尝试这个方法

所有时刻共用 13.82 µg/m³ 半宽，不能随预测的分散程度调整。
改用 `s=max((q95-q05)/2,1)` 作为尺度代理，在验证集计算 `abs(y-q50)/s` 的 90% 有限样本分位数 `k`，
新区间为 `[max(0,q50-k*s), q50+k*s]`。
这里借鉴标准化残差校准思路，用树百分位半宽作为尺度代理；不是把树百分位直接解释成真实浓度概率，
也没有训练分位数随机森林或额外的误差模型。

### 选择方案的过程：只用验证期

验证期按独立时间戳划成四块，前 1/4 初始化校准，依次预测后续三个时间块。
每块静态方案只用该块开始前已可获得的历史标签；要求历史样本特征时刻+3h早于块起点。
三个向前验证块共 12,492 条，不包含原测试集。
固定候选为全局绝对残差、分节点绝对残差、树间尺度标准化、树带加性扩张。
另外比较一个分节点最近 30 天的滚动方案，作为有标签持续回传条件下的研究方案，不参与静态方案选择。

| 方案 | 向前验证覆盖率 | 平均宽度 µg/m³ | 平均区间评分（越低越好） |
|---|---:|---:|---:|
| 固定全局半宽 | 89.25% | 25.98 | 46.22 |
| 分节点固定半宽 | 88.86% | 26.00 | 43.43 |
| 树间尺度标准化 | 89.47% | 24.97 | 35.25 |
| 树带加性扩张 | 89.63% | 23.31 | 35.94 |
| 分节点滚动 30 天 | 89.75% | 27.32 | 43.22 |

区间评分定义为 `width + 20*max(lower-y,0) + 20*max(y-upper,0)`，同时惩罚过宽和漏覆盖。
在四种静态校准方案中按汇总平均评分选出树间尺度标准化，没有根据原测试成绩挑方案。
滚动方案使用当时已经揭示的标签，而静态方案在每块起点冻结，部署条件不同。
不能仅凭滚动覆盖率高一点就声称它更好或已可实时部署；本实验假定目标在 3 小时后可用，尚未验证上游实际延迟。

### 冻结后复核已有测试预测

全验证集冻结得到 `k=1.5452243600022562`，尺度下限固定为 `1.0 µg/m³`。
固定后仅对选出的方案复核缓存测试预测，没有再尝试其它方案或按测试结果调 k。

| 方案 | test PICP | test MPIW µg/m³ |
|---|---:|---:|
| 原树间带 | 71.82% | 20.51 |
| 全局固定半宽校准 | 83.29% | 27.21 |
| 自适应宽度校准 | 88.09% | 31.74 |

相对固定半宽，覆盖率增加约 4.80 个百分点，同时平均宽度增加约 16.6%。
前向验证中自适应区间较窄，完整测试中却更宽，应分别报告，不能挑一个阶段的宽度做宣传。
测试分节点覆盖率：KL_City 84.63%、Port_Klang 91.49%、Sepang 88.16%。
仍未达到全局或逐节点 90%，不改变第 9 节关于时间依赖、验证集曾用于选参、事后扩展及 CAMS 目标的限制。

### 给 7 号的自适应版本接入

```python
import importlib
import json
from pathlib import Path
import numpy as np

R = importlib.import_module("05_reg_to_cls")
A = importlib.import_module("05_interval_adaptive")
cal = json.loads(Path("models/interval_calibration_adaptive.json").read_text(encoding="utf-8"))
tree_values = R.tree_predictions(rf_regressor, X)
quantiles = np.quantile(tree_values, [.05, .5, .95], axis=1).T
band = A.apply_calibration(quantiles, cal)  # 下界 / 中位数 / 上界
```

页面建议标签：“90% 目标覆盖率自适应区间；历史测试实测覆盖 88.09%”。
离线回放改读 `models/stage5_intervals_adaptive.npz`；模型、特征顺序和时刻含义沿用原接口。
更换回归模型后必须重新校准；不要把旧固定半宽的 JSON 传给自适应函数。
原 `metrics_stage5.json` 和两张图仍记录原始实验，不代表新版区间。

新产物：
- `src/05_interval_adaptive.py`、`tests/test_stage5_interval_adaptive.py`
- `models/interval_calibration_adaptive.json`（内嵌完整验证方案比较与选择记录）
- `models/metrics_stage5_interval_adaptive.json`
- `models/stage5_intervals_adaptive.npz`

当时的局部研究与应用脚本保留在本地，方案比较记录见原冻结参数中的选择证据。
2026-11-16 已将应用步骤整理为 `src/05_apply_adaptive_interval.py`，详见第 13 节。
纯计算模块不变；正式入口不依赖本地研究脚本或方案比较文件，已有结果禁止直接覆盖。

参考：[MAPIE 标准化残差分数](https://mapie.readthedocs.io/en/stable/content/conformal-prediction/conformity-scores/)。

## 11. 2026-11-14：Q5-3 接缝误差切片，保留官方 v1 成绩

按最新通告，Q5-1（用 v2 重跑 M5/M6）和 Q5-2 已取消。
本次仅完成 Q5-3；不改变官方模型、指标文件、特征表、分档阈值、benchmark 或区间方案。

### 输入与方法

- 官方 v1：`data/features_final.parquet`，SHA256 为 `64285f40e81c72ab4872d282d52be68401078e8c33dec50134d93e7877d2f55b`。
- `models/stage5_predictions.npz` 的 `reg_val/reg_test` 是 seed 42 的 M5 森林均值浓度预测。
- 使用 `models/ordinal_probs_val_test.npz` 的时间戳与节点标识核对行顺序，并核对标签。
- 连续浓度预测先于分档；M5 固定切点与 val 调整切点共享这份浓度预测，不涉及重新校准等级阈值。
- 按本地发出时刻切分：05–07 跨 08:00，17–19 跨 20:00，其他钟点为互斥第三组。
- 为严格落实组长的“跨 20:00 vs 其他时段”，另列非跨 20:00 的合计对照，它包含早间接缝窗口。

### 结果与限制

完整单表见报告 `reports/sec2_6_and_2_7.md` 的 §2.7 小节“Prediction Errors Around Forecast Product Seams”。
val：跨 20:00 MAE 12.2043，非跨 20:00 合计 5.1697 µg/m³，相差 7.0345，比例 2.3607。
test：跨 20:00 MAE 13.7345，非跨 20:00 合计 7.2617 µg/m³，相差 6.4729，比例 1.8914。
未做显著性检验或混杂调整；表中差异支持描述性关联，不单独证明接缝导致了全部额外误差。
这是已评估预测的事后切片，没有进行模型训练、新的测试预测或模型选择。

### 文件与复核

- `src/05_seam_error.py`：校验 v1 指纹、原缓存摘要和行标识，再汇总 val/test 的 MAE。
- `models/m5_seam_mae.csv`：8 行（两个切分各 4 行），含分组、样本数和未按展示位数截断的 MAE。
- `models/metrics_stage5_seam.json`：来源摘要、代码摘要、生成时间、Git 基线、分组含义及差值/比值。
- `tests/test_stage5_seam.py`：用合成数据检查窗口边界、加权合计、数组对齐及本地时间约定。
- `python -m unittest discover -s tests -p test_stage5_seam.py -v`
- `python src/05_seam_error.py`：在输出不存在的独立复现目录运行，已有输出会阻止覆盖。

原始 `models/metrics_stage5.json` 仍是官方成绩；新增的 `metrics_stage5_seam.json` 只存切片分析。
7 号合报告时引用新增表格即可，不需要替换主模型、主表或原区间结果。

## 12. 2026-11-15：§2.7 和 §2.8 英文报告交付

### 我产出了什么

- `reports/sec2_6_and_2_7.md` 由中文实验提纲更新为已确认的英文正文；文件名按报告拼接约定保留。
- §2.7 包含 M5 方法、固定与验证集调整阈值比较、M6 对照和 08:00/20:00 接缝误差表。
- §2.8 包含树间百分位区间、残差校准、尺度与区间公式、覆盖率和平均宽度对照、节点差异及限制。
- 正文与本次确认的上一版复核稿一致；不加入随后建议的 Y 定义或 n=16,656 补充句。

### 字段和接口说明

正文是 Markdown，三张表和公式随正文提供。浓度及浓度误差单位为 µg/m³，等级编号为 0–3。
模型、数据、概率矩阵、分档阈值、已有预测与运行接口均沿用第 1–11 节。

### 关键决策和理由

- 沿用官方 v1 M5/M6 成绩，不用 v2 对照覆盖 `metrics_stage5.json`。
- M6 保留为附录对照，不改主模型决定。
- 区间采用第 10 节自适应方案：k=1.54522436，测试覆盖 88.09%，平均宽度 31.74 µg/m³。
- 接缝表复用本分支现有 Q5-3 结果；不重新训练或生成测试预测。

### 已知缺陷和坑

- 90% 是目标覆盖率；90.01% 为完整验证集的样本内校准检查，88.09% 才是本方案的测试覆盖率。
- 验证集曾用于选模型，数据有时空依赖；本区间分析为事后扩展，不提供逐样本或未来 90% 保证。
- 三张表由总报告统一编号。接缝表最后一行与第一、第三行重叠，不能重复计数。
- 数字自查会列出比例、差值、百分比及 k 的显示截断，需按来源人工复核；文字检查不证明原创性。

### 下一棒怎么用

7 号直接从 `reports/sec2_6_and_2_7.md` 读取 §2.7 与 §2.8，按总报告顺序拼接，保留三张表和公式。
Frank and Hall 与 MAPIE 的来源已在前文交接中列出，请并入总参考文献表。
校验命令：`python tools/check_prose.py reports/`、`python tools/check_report_numbers.py reports/sec2_6_and_2_7.md`、`python tools/check_merge_gate.py fix/05-seam-slices`。

### 我没做的事

没有修改模型、官方指标、特征表、区间产物、benchmark、README、CONTRIBUTING、共享配置或其他成员报告。
没有重跑 M5/M6、继续调 M6、追加变量说明，或生成新 Word。

## 13. 2026-11-16：自适应区间入口交付时的记录（重评现状见第 14 节）

### 本次交付

- `src/05_apply_adaptive_interval.py`：读取冻结参数并应用到缓存树百分位，不拟合参数或调用模型预测。
- `tests/test_stage5_apply_adaptive.py`：核对看板接口、防覆盖、数据指纹、校准样本数和行标识。
- `docs/qc_fix/stage5_adaptive_entrypoint.md`：新旧数据状态、旧结果复现、后续依赖和对照表。
- 报告 §2.8 补充复现入口与数据版本界限，保留原有英文正文及历史指标。
- `src/05_interval_adaptive.py`、官方指标、模型、预测缓存和冻结参数均未修改。

### 输入与输出

运行目录为仓库根目录：

```bash
python -m unittest discover -s tests -p "test_stage5*.py" -v
python src/05_apply_adaptive_interval.py
```

主要输入为 `models/stage5_predictions.npz` 和 `models/interval_calibration_adaptive.json`。
浓度真值从 `data/features_final.parquet` 读取，特征清单为 `models/selected_features.json`。
`models/ordinal_probs_val_test.npz` 提供时间戳和节点行标识，`metrics_stage5.json` 提供来源指纹。
注意该概率文件的 `y_val/y_test` 是等级标签，不能拿来计算浓度区间覆盖率。

仅写出 `models/metrics_stage5_interval_adaptive.json` 和 `models/stage5_intervals_adaptive.npz`。
冻结参数 JSON 只读，正式入口不重新选择方法、放大倍数、下限或模型。
任一输出已存在就抛出 `FileExistsError`；需先保留旧文件后再运行，不提供覆盖选项。

JSON 顶层保留 `method`、`nominal_coverage`、`scale_multiplier`、`scale_floor`、
`calibration_sha256`、`limitations`、`new_model_fits`、`new_model_test_predictions`、`unit`，
以及原有 `intervals_sha256`。`val/test` 各含 `overall` 与 `by_node`，每组指标均为
`PICP`、`MPIW`、`n_samples`，节点键为 `KL_City`、`Port_Klang`、`Sepang`。

NPZ 仅含六个键：`interval_val`、`ts_val`、`station_id_val`、
`interval_test`、`ts_test`、`station_id_test`；区间三列顺序是下界、中位数、上界。
时间戳仍是本地发出时刻，目标时刻加 3 小时。

### 已验证与待接续

旧输入的完整独立复现通过：val/test 样本数 16,656 / 16,647，
`interval_val.shape=(16656, 3)`；所有总体、分节点指标和六个数组与旧产物完全相等。
这项检查只证明旧结果可复现，不是 QC 重评成绩。

main@7a46440 的新表已实读通过：111,038 × 32，包含 `gran_low_12h`，
train/val/test 为 77,744 / 16,647 / 16,647。
当前 main 的 stage-05 缓存和冻结参数仍绑定旧表。新表混用旧缓存的检查按预期拒跑，未写输出。

下一步需要与新表对应的 stage-05 预测、行标识、来源登记，以及按原方法得到并冻结的校准参数。
仅按行号删掉旧验证集 9 行不能替代新数据重评；测试集行数相同也不能证明特征和预测相同。
4 号的 M2/M3/M4 重训不会自动生成上述 M5 树百分位缓存。需由组长明确 stage-05 上游重评的安排。
对应输入到齐后，保留旧产物，再运行入口，核对两组样本数和两组形状均为 16,647 / (16647, 3)，
填入新旧 PICP/MPIW 对照并说明宽窄变化；不能用旧 87.97% / 32.11 冒充新表成绩。

## 14. 2026-11-16：M5 QC 重评与新版区间已完成

### 我产出了什么

- `src/05_refit_m5_qc.py`：固定原最终配方，重训三个 M5 种子、仅给 seed 42 生成一次测试预测；拟合并冻结新验证集校准参数。
- `models/metrics_stage5.json`、`rf_regressor.joblib`、`stage5_selection.json`、`stage5_qc_refit_plan.json` 与两份预测缓存：M5 已切到 QC 表。
- 同名自适应/固定宽度校准 JSON、指标 JSON 和区间 NPZ：均对应新版 M5；正式应用入口只读参数、只应用缓存，不拟合。
- `models/m5_seam_mae.csv` 与 `metrics_stage5_seam.json`、两张阶段 5 图：从新版缓存生成，没有第二轮测试预测。
- `fig_interval_band.png` 当前展示自适应校准区间，标题注明 90% 是目标、测试实测覆盖 88.1%；原始树百分位只作为指标对照保留。
- `models/metrics_stage5_qc_comparison.json`、`docs/qc_fix/stage5_refit_comparison.md`：新旧成绩、区间宽窄与超过 0.005 变化的逐项说明、模型 SHA256。
- `docs/qc_fix/stage5_benchmark_rows.csv`：给 6 号更新 benchmark 的两条 M5 记录，含特征表指纹；没有修改共享 benchmark。
- `docs/qc_fix/pre_refit_stage5/`：保留旧指标、选择记录、预测缓存、校准与区间、接缝表；旧模型可从 `48ef4d2` 恢复，指纹另列。

### 字段与接口

新版数据 SHA256 为 `e0f27fe33847f170cfb7b7690afed56407ea66cc7247ad3c9172eed9f517e8de`。
train/val/test 为 77,744 / 16,647 / 16,647，区间验收输出为：

```text
val.n_samples=16647 / test.n_samples=16647
interval_val.shape=(16647, 3) / interval_test.shape=(16647, 3)
```

看板的 JSON 层级和 NPZ 六键不变。`new_model_fits=0`、`new_model_test_predictions=0` 指区间应用步骤；
M5 重训在主指标中另记为 3 次拟合、1 批测试树预测，M6 均为 0。
`ordinal_probs_val_test.npz` 的 M6 概率来自旧模型缓存，按精确的时间戳/节点键连接并验证标签未变，
不是按位置删九行，也不是重新预测。该文件两段现在各 16,647 行，便于和新版 M5/区间按键对齐。

### 关键决策和理由

- 不重复候选搜索、CV 或分档阈值搜索。沿用 100 棵树、深度 14、叶节点下限 10、max_features=0.5；阈值 17/33/55。
- 三个种子仍为 42/43/44，只在验证集做稳定性检查；最终模型沿用 seed 42。
- 自适应方法、名义覆盖率 0.90 和 scale_floor=1 不变；只在新版 val 按原次序统计量重估放大倍数 k=1.5474406353，然后冻结。
- 新旧共同样本的目标完全相同；选择特征用 hour_cos 替换 pm2_5_mean_6h，train/val 各少 9 行，测试中 far_source_pm25_mean_24h 有 1,149 行改变。多项输入变化同时发生，不能把全部增益归因于 hour_cos。
- M6 保留历史附录，不拟合、不产生新的模型预测。`models.M6_Ordinal_RF.qc_refit=false`，其旧训练指纹和特征顺序独立登记。

### 已知缺陷和坑

- 测试目标早已看过，本次明确记作数据修复后的重评，不称为新的盲测。
- 新 M5 测试六项指标为 0.7104 / 0.7139 / 0.8789 / 0.2854 / 0.0091 / 0.7046；高危召回没有明显提升，不能只报 F1。
- 自适应测试覆盖 88.09%，仍低于 90%；KL_City 为 84.63%，仍是主要覆盖缺口。平均宽度从 32.11 降到 31.74 µg/m³。
- M6 历史模型的 `selected_features_` 与新版 Top-25 不同，不能把新矩阵直接喂给它；6 号的解释/泛化若调用模型，必须明确使用哪个版本。
- M6/M7 的缓存概率仍是历史结果，不能称其已在 QC 表上重训，也不能据此重新作同表 M5/M6 优劣结论。
- 共享 `all_models_benchmark.csv`、总报告、README.md 与 app.py 的静态文字由相应负责人更新；本次未越权改动。
- 指标中的 `training_code_sha256` 保留实际执行版本；训练后纠正了参数记录的源模块指针与文本换行，模型、预测、区间数组和数值均未改变。

### 下一棒怎么用

1. 6 号从 `stage5_benchmark_rows.csv` 取两行新 M5 成绩，按团队最终版本口径更新共享 benchmark；不要连带把 M6/M7 标成已重训。
2. 7 号继续读同名缓存；区间标签使用验证 90.01% / 测试 88.09% / 宽度 31.74。§2.7 的原 M5/M6 比较明确标为历史实验，新增 QC 对照、接缝表与 §2.8 使用新结果。
3. `python tools/verify_stage5.py` 只核对缓存、冻结配方、区间和历史 M6 对齐，不调用模型预测；重复运行训练或应用入口会拒绝覆盖。

独立复现需使用干净目录，不能对已经完成的产物再次运行训练。基线 `48ef4d2` 含旧缓存与新 QC 表：

```bash
git worktree add ../HazeGuard-KL_m5_qc_reproduce 48ef4d23ccf8b0c4c494eee749383c158da38815
cd ../HazeGuard-KL_m5_qc_reproduce
# 在独立目录中从本次交付分支取下面的入口，再运行；不要取 models/ 覆盖旧输入。
git checkout origin/fix/05-adaptive-entrypoint -- src/05_refit_m5_qc.py src/05_apply_adaptive_interval.py src/05_ordinal_rf.py src/05_seam_error.py tools/compare_stage5_qc.py tools/verify_stage5.py
python src/05_refit_m5_qc.py --n-jobs 8
python tools/compare_stage5_qc.py
python tools/verify_stage5.py
```

### 我没做的事

没有重新调参、重选分档阈值、再比较校准方案、按测试覆盖率凑 90%、重训或优化 M6。
没有改共享配置、指标定义、数据表、01/04/06/07 代码、共享 benchmark 或生成 Word。
