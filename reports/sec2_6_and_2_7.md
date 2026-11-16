## 2.7 Regression Based Risk Classification and Ordinal Comparison

### M5 Regression Model

M5 employs a Random Forest regressor to predict PM2.5 concentrations three hours ahead. The model first averages the concentration predictions from individual trees, then converts this value into one of four risk levels. The thresholds for the true labels are 15, 35, and 55 µg/m³. This can be understood as “first predicting the specific concentration, then determining which tier it belongs to.” During training, the numerical differences between concentration values are preserved.

The official M5 results are based on the v1 feature set, using 25 filtered features and three receptor grid points: KL_City, Port_Klang, and Sepang. After temporal splitting, the training set contains 77,753 samples, the validation set 16,656, and the test set 16,647. The prediction targets are derived from CAMS atmospheric model products, not from ground-based monitoring station measurements. The v1 feature set includes hour_sin but lacks hour_cos; this time encoding issue has been documented. The corrected v2 feature set was evaluated separately by M8 and cannot be used to replace M5’s official results.

The selected Random Forest consists of 100 trees with a maximum depth of 14, a minimum of 10 samples per leaf node, and considers half of the features at each split. Parameter tuning employed five-fold cross-validation with temporal ordering. Samples from different grid points at the same timestamp were kept on the same side of the fold split. If a training sample’s three-hour-ahead target fell into a subsequent validation interval, that training sample was removed to avoid target time overlap. Since feature selection was performed on the entire training set, this internal cross-validation does not constitute full nested validation.

### Fixed and Validation Tuned Thresholds

M5 compared two sets of prediction binning thresholds: fixed thresholds of 15, 35, and 55, and optimized thresholds of 17, 33, and 55 µg/m³ selected on the validation set. The true labels were always partitioned according to 15, 35, and 55. Only the rule for converting predicted concentrations into risk levels was changed; the true labels remained unmodified. Thresholds were first selected on the validation set before evaluating the test set. Stability checks were conducted using seeds 42, 43, and 44, while the final test results were obtained using the pre-selected seed 42.

| M5 decision boundaries | Test Macro-F1 | High-risk recall | MAOE | Severe jumping error rate |
|---|---:|---:|---:|---:|
| Fixed | 0.7000 | 0.8413 | 0.2787 | 0.0100 |
| Validation-tuned | 0.7078 | 0.8788 | 0.2873 | 0.0091 |

High-risk recall measures the proportion of samples that truly belong to Level 2 or Level 3 and are predicted as one of these two high-risk levels. MAOE is the mean absolute error between predicted and true levels; severe jumps refer to discrepancies of at least two levels. After threshold adjustment, Macro F1 and high-risk recall improved, and severe jumps decreased, but MAOE increased from 0.2787 to 0.2873. The threshold adjustment did not improve all metrics.

### M6 Ordinal Classification Comparison

M6 adopts the cumulative decomposition approach proposed by Frank and Hall, training three binary-classification random forests to estimate P(Y > 0), P(Y > 1), and P(Y > 2), respectively. Each model uses balanced_subsample class weights. In implementation, the three cumulative probabilities are first adjusted to ensure a non-increasing order, and then the probabilities for the four levels are recovered. This step resolves inconsistencies in probability ordering but does not establish probability calibration.

For M6, the test Macro F1 was 0.6839, high-risk recall was 0.8642, MAOE was 0.3169, and the severe jump error rate was 0.0239. The threshold-adjusted M5 outperformed M6 on all four metrics, with consistent ranking on the validation set. However, M6 had a lower average asymmetric cost of 0.6616 compared to 0.7083 for M5; M6 was therefore not worse across all evaluation criteria. The project retained M5 as the primary solution, while placing M6 in the ablation appendix. This conclusion applies only to the current implementation and dataset and should not be generalized to claim that ordinal classification methods are universally ineffective.

### Prediction Errors Around Forecast Product Seams

This analysis reused previously saved concentration predictions from M5 (seed 42) without retraining the model or generating new predictions. Predictions were grouped by their issuance time in Kuala Lumpur local time: forecasts issued between 05:00–07:00 correspond to target times 08:00–10:00, and those issued between 17:00–19:00 correspond to target times 20:00–22:00. These two windows each span the forecast product seams recorded in §2.1.5 at 08:00 and 20:00, respectively.

| Local issue-time group | Validation n | Validation MAE (µg/m³) | Test n | Test MAE (µg/m³) |
|---|---:|---:|---:|---:|
| 05:00–07:00: crosses 08:00 | 2,088 | 6.2956 | 2,079 | 10.7498 |
| 17:00–19:00: crosses 20:00 | 2,079 | 12.2043 | 2,082 | 13.7345 |
| Remaining hours, excluding both windows | 12,489 | 4.9815 | 12,486 | 6.6809 |
| All hours outside the 20:00 window | 14,577 | 5.1697 | 14,565 | 7.2617 |

MAE is the mean absolute error of concentration predictions, measured in µg/m³. The first three rows are mutually exclusive and cover all samples in the dataset. The last row is a sample-size-weighted combination of the first and third rows, used to directly answer “how does the 20:00 seam compare with other periods.” The last row overlaps with the first and third rows and is not an additional sample group.

In the test set, the MAE for the 20:00 window is 1.8914 times that of other periods, exceeding them by 6.4729 µg/m³; the corresponding ratio in the validation set is 2.3607. Errors in the morning window are also higher than those in the remaining periods after excluding the two windows. Both datasets exhibit the phenomenon of “larger errors in these windows.” This is a descriptive comparison that does not control for concentration levels, seasonality, or correlations among samples, nor does it perform significance testing. This comparison does not establish that all additional error was caused by the seam.

## 2.8 Calibration of Concentration Prediction Intervals

### Original Tree Based Prediction Intervals

The original intervals were obtained by directly taking the 5th and 95th percentiles across individual tree predictions for each sample. This reflects the range of disagreement among trees. On the test set, they covered only 71.83% of the true targets, with an average width of 20.79 µg/m³. “Covering the middle 90% of tree predictions” is not equivalent to “covering 90% of true concentrations.” These two quantities describe different forms of coverage.

### Residual Based Interval Calibration

The first method centers the interval on the median of tree predictions and determines a uniform half-width using absolute residuals from the validation set. The second method allows the interval half-width to vary with the degree of disagreement among tree predictions, employing standardized residuals. The percentile span of tree predictions serves merely as a proxy for scale, without training an additional residual prediction model. The MAPIE conformity score documentation describes such standardized residual methods.

For the i-th sample, let m_i denote the median of tree predictions, and let q_0.05,i and q_0.95,i represent the 5th and 95th percentiles of tree predictions. The scale s_i is half the percentile span, with a lower bound of 1 µg/m³ to prevent excessively small scales.

\[
s_i=\max((q_{0.95,i}-q_{0.05,i})/2,1\;\text{µg/m³})
\]

The standardized absolute residual on the validation set is defined as r_i = |y_i − m_i| / s_i, where y_i is the target concentration. The target coverage rate is set to 0.90. Given n validation samples, sort the residuals in ascending order and take the ceil((n + 1) × 0.90)-th value, without interpolation. Here, ceil denotes rounding up. The amplification factor recorded is k = 1.54522436, yielding intervals with the lower bound truncated at 0 to avoid negative concentrations.

\[
[\max(0,m_i-k s_i),m_i+k s_i]
\]

Samples with greater disagreement among tree predictions typically receive wider intervals; those with smaller disagreement typically receive narrower intervals.

### Calibration Method Selection and Results

Four static calibration methods were compared experimentally, using three temporally advancing validation blocks. The selection criterion was the mean interval score, which penalizes both overly wide intervals and cases where the true target falls outside the interval. Methods and parameters were fixed first, and then interval coverage and width were computed for the saved test predictions. Calibration did not alter the regression model, risk stratification thresholds, or classification performance.

| Interval method | Test coverage | Mean test width (µg/m³) |
|---|---:|---:|
| Original tree-percentile band | 71.83% | 20.79 |
| Common half-width calibration | 83.41% | 27.43 |
| Adaptive-width calibration | 87.97% | 32.11 |

The adaptive method improved coverage by 4.56 percentage points over the uniform half-width method, increasing the average interval width by approximately 17.1%. This represents a trade-off between coverage and interval width. The calibrated coverage on the full validation set was 90.01%, reflecting evaluation on the data used for calibration. Test coverage remained below the 90% target. Differences across nodes were observed: KL_City achieved 84.77%, Port_Klang 91.19%, and Sepang 87.96%.

The validation set had also been used for model selection, and the hourly samples have temporal and spatial dependence. The interval study was a post-hoc extension after the original test results were known. The 90% target does not guarantee future coverage or coverage for an individual forecast.

### Reproduction Entry and Data Version

The interval results above belong to the original M5 prediction cache. The application entry is `python src/05_apply_adaptive_interval.py`. It reads frozen adaptive parameters and saved tree percentiles, checks their data fingerprints and row identities, and writes the dashboard's existing JSON and NPZ interfaces. It does not fit models or select calibration parameters. Both output files must be archived before running; existing files are never overwritten.

The QC feature table promoted in commit `7a46440` has 16,647 validation rows. The saved M5 cache and calibration still refer to 16,656 validation rows. Applying the new entry to the original inputs reproduces all reported interval scores and all six saved arrays exactly. Combining the QC table with the original cache is rejected before any output is written. The results above have not been re-evaluated on the QC table, and no change in interval width can yet be attributed to that repair.
