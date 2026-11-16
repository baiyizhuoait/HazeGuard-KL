# M5 QC 重评：同配方新旧对照

原模型参数、分档切点与三个种子不变；新表含 77,744 / 16,647 / 16,647 行。
M6 未重训，原模型与原预测保留为历史附录；本表不把它当作 QC 重训结果。

## M5 分类指标

| 模型 | 切分 | 指标 | 旧 | QC | 变化 |
|---|---|---|---:|---:|---:|
| M5_fixed | val_seed42 | Macro_F1 | 0.7160 | 0.7194 | +0.0034 |
| M5_fixed | val_seed42 | Balanced_Accuracy | 0.6991 | 0.7015 | +0.0024 |
| M5_fixed | val_seed42 | High_Risk_Recall | 0.8092 | 0.8081 | -0.0011 |
| M5_fixed | val_seed42 | MAOE | 0.2568 | 0.2524 | -0.0044 |
| M5_fixed | val_seed42 | Severe_Jumping_Error_Rate | 0.0070 | 0.0060 | -0.0010 |
| M5_fixed | val_seed42 | Avg_Asymmetric_Cost | 0.5929 | 0.5883 | -0.0046 |
| M5_fixed | test_seed42 | Macro_F1 | 0.7000 | 0.7046 | +0.0046 |
| M5_fixed | test_seed42 | Balanced_Accuracy | 0.6793 | 0.6836 | +0.0043 |
| M5_fixed | test_seed42 | High_Risk_Recall | 0.8413 | 0.8419 | +0.0006 |
| M5_fixed | test_seed42 | MAOE | 0.2787 | 0.2750 | -0.0037 |
| M5_fixed | test_seed42 | Severe_Jumping_Error_Rate | 0.0100 | 0.0100 | +0.0000 |
| M5_fixed | test_seed42 | Avg_Asymmetric_Cost | 0.7242 | 0.7180 | -0.0062 |
| M5_fixed | val_three_seed_mean | Macro_F1 | 0.7152 | 0.7178 | +0.0026 |
| M5_fixed | val_three_seed_mean | Balanced_Accuracy | 0.6977 | 0.7000 | +0.0023 |
| M5_fixed | val_three_seed_mean | High_Risk_Recall | 0.8094 | 0.8081 | -0.0013 |
| M5_fixed | val_three_seed_mean | MAOE | 0.2566 | 0.2542 | -0.0024 |
| M5_fixed | val_three_seed_mean | Severe_Jumping_Error_Rate | 0.0066 | 0.0063 | -0.0003 |
| M5_fixed | val_three_seed_mean | Avg_Asymmetric_Cost | 0.5936 | 0.5906 | -0.0029 |
| M5_val_tuned | val_seed42 | Macro_F1 | 0.7214 | 0.7251 | +0.0037 |
| M5_val_tuned | val_seed42 | Balanced_Accuracy | 0.7258 | 0.7285 | +0.0027 |
| M5_val_tuned | val_seed42 | High_Risk_Recall | 0.8612 | 0.8627 | +0.0015 |
| M5_val_tuned | val_seed42 | MAOE | 0.2628 | 0.2582 | -0.0046 |
| M5_val_tuned | val_seed42 | Severe_Jumping_Error_Rate | 0.0062 | 0.0055 | -0.0007 |
| M5_val_tuned | val_seed42 | Avg_Asymmetric_Cost | 0.5828 | 0.5752 | -0.0076 |
| M5_val_tuned | test_seed42 | Macro_F1 | 0.7078 | 0.7104 | +0.0026 |
| M5_val_tuned | test_seed42 | Balanced_Accuracy | 0.7117 | 0.7139 | +0.0022 |
| M5_val_tuned | test_seed42 | High_Risk_Recall | 0.8788 | 0.8789 | +0.0001 |
| M5_val_tuned | test_seed42 | MAOE | 0.2873 | 0.2854 | -0.0019 |
| M5_val_tuned | test_seed42 | Severe_Jumping_Error_Rate | 0.0091 | 0.0091 | +0.0000 |
| M5_val_tuned | test_seed42 | Avg_Asymmetric_Cost | 0.7083 | 0.7046 | -0.0037 |
| M5_val_tuned | val_three_seed_mean | Macro_F1 | 0.7195 | 0.7244 | +0.0049 |
| M5_val_tuned | val_three_seed_mean | Balanced_Accuracy | 0.7234 | 0.7279 | +0.0045 |
| M5_val_tuned | val_three_seed_mean | High_Risk_Recall | 0.8600 | 0.8619 | +0.0019 |
| M5_val_tuned | val_three_seed_mean | MAOE | 0.2644 | 0.2588 | -0.0056 |
| M5_val_tuned | val_three_seed_mean | Severe_Jumping_Error_Rate | 0.0061 | 0.0057 | -0.0004 |
| M5_val_tuned | val_three_seed_mean | Avg_Asymmetric_Cost | 0.5871 | 0.5772 | -0.0099 |

## 总体区间指标

| 方法 | 切分 | 旧 PICP | QC PICP | 旧 MPIW | QC MPIW |
|---|---|---:|---:|---:|---:|
| adaptive | val | 0.90009606 | 0.90010212 | 25.44245539 | 25.18841406 |
| adaptive | test | 0.87973809 | 0.88093951 | 32.11199106 | 31.73574759 |
| fixed | val | 0.90009606 | 0.90010212 | 27.08278526 | 26.86661841 |
| fixed | test | 0.83414429 | 0.83294287 | 27.42551744 | 27.20768675 |

## 绝对变化超过 0.005 的逐项说明

下列说明界定原因范围，不把联合数据变更的增益单独归因于某一列。

- M5_fixed / test_seed42 / Avg_Asymmetric_Cost：-0.0062。新表更换一个选择特征并冲刷边界行，随后按原超参重新拟合；未新增调参或按测试结果选择方案。
- M5_val_tuned / val_seed42 / Avg_Asymmetric_Cost：-0.0076。新表更换一个选择特征并冲刷边界行，随后按原超参重新拟合；未新增调参或按测试结果选择方案。
- M5_val_tuned / val_three_seed_mean / MAOE：-0.0056。新表更换一个选择特征并冲刷边界行，随后按原超参重新拟合；未新增调参或按测试结果选择方案。
- M5_val_tuned / val_three_seed_mean / Avg_Asymmetric_Cost：-0.0099。新表更换一个选择特征并冲刷边界行，随后按原超参重新拟合；未新增调参或按测试结果选择方案。
- adaptive / val / overall / MPIW：-0.254041。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- adaptive / val / KL_City / MPIW：-0.350163。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- adaptive / val / Port_Klang / PICP：+0.006987。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- adaptive / val / Sepang / MPIW：-0.410533。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- adaptive / test / overall / MPIW：-0.376243。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- adaptive / test / KL_City / MPIW：-0.628061。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- adaptive / test / Port_Klang / MPIW：-0.223750。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- adaptive / test / Sepang / MPIW：-0.276920。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- fixed / val / overall / MPIW：-0.216167。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- fixed / val / KL_City / MPIW：-0.227549。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- fixed / val / Port_Klang / MPIW：-0.185665。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- fixed / val / Sepang / MPIW：-0.235286。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- fixed / test / overall / MPIW：-0.217831。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- fixed / test / KL_City / MPIW：-0.227705。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- fixed / test / Port_Klang / MPIW：-0.202360。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。
- fixed / test / Sepang / MPIW：-0.223428。M5 按新输入重拟合后树预测改变，区间参数按原规则仅在新验证集重估，区间及覆盖随之变化；未按测试覆盖率调整。

## 指纹

- `rf_regressor.joblib`：`759beafe26d8bf13d5e27d1577d9af4227a471ad11060420a68353a3a4afbf5f`。
- `ordinal_rf.joblib`：`8a1c16807a6c3b930fbdc800b50aee6ddc9cea6b92f6e7ccc9e9801480744172`。

## 给 6/7 号

`stage5_benchmark_rows.csv` 给出两条 M5 测试记录和特征表指纹，供 6 号更新 benchmark；它不是共享 CSV 的替代品。
M6/M7 仍使用历史概率：新 NPZ 只是按精确键对齐，不能宣称它们已经在 QC 特征表上重训。
旧 ordinal_rf.joblib 的 selected_features_ 仍是旧顺序，不能直接传入新版 Top-25 矩阵。
看板区间继续读同名自适应 JSON/NPZ，字段与键名不变。
