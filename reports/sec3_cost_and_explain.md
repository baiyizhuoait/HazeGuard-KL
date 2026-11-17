# Section 3: Cost-Sensitive Optimization, Model Interpretability and Generalization

## 3.1 Cost-Sensitive Threshold Optimization (M7)

### 3.1.1 Objective

The HazeGuard-KL system predicts four ordinal haze-risk levels (Level 0–3). Since underestimating severe haze pollution may result in delayed warnings, the project uses an asymmetric cost matrix that assigns higher penalties to under-prediction than over-prediction.

The objective of M7 is to reduce the Average Asymmetric Cost through decision-threshold optimization while preserving the original trained model.

### 3.1.2 Methodology

A three-dimensional grid search was implemented over three ordinal decision thresholds:

\[
\tau = (\tau_1,\tau_2,\tau_3)
\]

Each threshold was searched from 0.05 to 0.95 with a step size of 0.05, resulting in:

\[
19^3 = 6859
\]

candidate threshold combinations.

The optimization objective was:

\[
\min_{\tau} \frac{1}{N}\sum_{i=1}^{N}C[y_i,\hat{y}_i(\tau)]
\]

where \(C\) is the project's predefined asymmetric cost matrix.

Threshold selection was performed exclusively on the validation set containing 16,647 samples. No GPSO or evolutionary optimization algorithm was used.

### 3.1.3 Experimental Results

The optimal thresholds were:

- τ₁ = 0.20
- τ₂ = 0.30
- τ₃ = 0.45

The validation-set results are summarized below.

| Metric | Result |
|---|---:|
| Default Average Asymmetric Cost | 0.5620 |
| M7 Grid Search Cost | 0.4495 |
| Bayesian Decision Cost | 0.4890 |
| Macro-F1 | 0.6884 |
| Balanced Accuracy | 0.7143 |
| High-Risk Recall | 0.9360 |
| MAOE | 0.3100 |
| Severe Jumping Error Rate | 0.0162 |

Compared with the default thresholds, grid search reduced the validation asymmetric cost by approximately 20.01%.

The achieved High-Risk Recall of 93.60% indicates that the optimized decision thresholds identified a high proportion of actual Level 2/3 cases.

However, cost-sensitive optimization involves trade-offs. Improvements in high-risk detection do not necessarily imply improvements in all classification metrics.

### 3.1.4 Bayesian Decision Comparison

A Bayesian minimum expected cost decision rule was used as a theoretical comparison:

\[
\hat y = \arg\min_j \sum_i P(Y=i\mid x)C[i,j]
\]

The Bayesian method achieved an empirical validation cost of 0.4890, compared with 0.4495 for grid search.

This comparison should not be interpreted as evidence that grid search is theoretically superior to Bayesian decision-making. Differences may be influenced by model probability calibration and the specific validation distribution.

The final thresholds were saved in `models/cost_thresholds.json`.

### 3.1.5 Independent Test-Set Evaluation and Performance Trade-offs

After selecting the optimal thresholds on the validation set, the M7 Grid Search decision rule was evaluated once on the independent test set containing 16,647 samples. The selected thresholds were **[0.20, 0.30, 0.45]** and were fixed before test evaluation. No threshold tuning or model selection was performed using the test labels.

Three decision strategies were compared using the same test-set probability predictions: the default decision rule, the M7 cost-sensitive grid-search thresholds, and the Bayesian minimum expected cost decision rule.

**Table: Independent test-set performance comparison**

| Metric | Default | M7 Grid Search | Bayesian |
|---|---:|---:|---:|
| Macro-F1 | 0.6972 | 0.6659 | 0.6247 |
| Balanced Accuracy | 0.7192 | 0.6656 | 0.6524 |
| High-Risk Recall | 0.8737 | **0.9376** | 0.9594 |
| MAOE | 0.3013 | 0.3300 | 0.3993 |
| Severe Jumping Error Rate | 0.0171 | 0.0173 | 0.0320 |
| Avg Asymmetric Cost | 0.6497 | **0.5405** | 0.5458 |

**High-risk recall and asymmetric cost improvement.**

Compared with the default decision rule, M7 increased High-Risk Recall from **0.8737 to 0.9376**, an improvement of **6.39 percentage points**. At the same time, Avg Asymmetric Cost decreased from **0.6497 to 0.5405**, representing a relative reduction of approximately **16.81%**.

These improvements indicate that the validation-selected cost-sensitive thresholds successfully transferred to the independent test set. M7 identified a larger proportion of genuinely high-risk haze cases while reducing the average penalty defined by the project's asymmetric cost matrix.

**Macro-F1 deterioration and performance trade-off.**

The improvement in high-risk detection was not achieved without sacrificing other classification metrics. In particular, **Macro-F1 decreased from 0.6972 to 0.6659**, an absolute reduction of **0.0313**. Balanced Accuracy also decreased from 0.7192 to 0.6656, while MAOE increased from 0.3013 to 0.3300. The Severe Jumping Error Rate rose slightly from 0.0171 to 0.0173.

These results demonstrate that M7 is not universally superior to the default classifier. Its threshold adjustment changes the balance between missed hazardous events and precautionary warnings, prioritizing the project's asymmetric cost objective rather than overall class-balanced classification performance.

**Justification based on asymmetric misclassification costs.**

The project's cost matrix assigns substantially higher penalties to certain dangerous underprediction errors than to many overprediction errors. For example, predicting Level 0 when the true condition is Level 2 incurs **8 cost units**, while predicting Level 0 when the true condition is Level 3 incurs **15 cost units**. By comparison, several less severe errors incur penalties of only 1–5 cost units.

Therefore, a reduction in Macro-F1 may be acceptable when it accompanies fewer costly high-risk misses and a meaningful reduction in expected operational loss. The observed trade-off is consistent with a precautionary haze-warning application in which missing severe pollution conditions can have greater consequences than issuing some additional warnings.

Nevertheless, this justification depends on the validity of the specified cost matrix. M7 should be considered a **cost-sensitive operational alternative**, rather than a classifier that dominates the default approach on every metric.

**Comparison with the Bayesian decision rule.**

The Bayesian minimum expected cost strategy achieved the highest High-Risk Recall of 0.9594, but its test-set Avg Asymmetric Cost was 0.5458, slightly higher than M7's 0.5405. It also produced a lower Macro-F1 of 0.6247, a higher MAOE of 0.3993, and a higher Severe Jumping Error Rate of 0.0320.

This comparison provides a useful theoretical decision-making reference. Although the Bayesian rule minimizes expected cost for each probability vector under the specified cost matrix, its empirical performance depends on the quality and calibration of those probabilities. Consequently, it is not guaranteed to achieve the lowest realized cost on a finite test set.

**Conclusion.**

The independent test evaluation supports the use of M7 when the primary objective is to reduce the operational consequences of missed high-risk haze events. Relative to the default decision rule, M7 improved High-Risk Recall by **6.39 percentage points** and reduced Avg Asymmetric Cost by approximately **16.81%**, while lowering Macro-F1 by **0.0313**.

The result should therefore be presented as an explicit trade-off: **better high-risk protection and lower asymmetric cost at the expense of some general classification performance**. All M7 thresholds were selected using validation data only, and the independent test results were used exclusively for final evaluation.

*Results source: `models/m7_test_metrics.json`; threshold-selection record: `models/cost_thresholds.json`.*

## 3.2 Model Interpretability

### 3.2.1 Permutation Feature Importance

Permutation importance was applied to the M6 Ordinal Random Forest using validation observations. The experiment sampled 1,000 validation observations and performed five permutation repetitions, with Macro-F1 as the scoring metric.

The M6 model was not retrained following the upstream quality-control (QC) update. Instead, the existing fitted model and saved predictions were retained. The updated validation data contained 16,647 observations, which were aligned with the saved predictions using exact station identifiers and timestamps.

To ensure compatibility with the original fitted model, the permutation experiment used the historical feature table from Git commit `67e8977` and the 25 training features stored in the model's `selected_features_` attribute. This was necessary because the latest QC feature selection differed from the original M6 training configuration. The original model predictions were successfully reproduced, with a maximum probability difference of approximately 3.33 × 10⁻¹⁶. This verification was completed before generating the feature-importance results.

The analysis identified `pm2_5` as the most influential feature, with a mean permutation importance of 0.2569, followed by `hour_sin` at 0.0809. Other influential predictors included `no2_lag_6` (0.0172), `gran_slope_12h` (0.0099), and `no2_lag_12` (0.0091).

These findings indicate that the original M6 model relies strongly on current PM2.5 concentration and temporal information when predicting future haze-risk levels. Several lagged pollution and trend features also contribute to prediction performance.

The comparatively low individual importance of some spatial and upstream features should not be interpreted as evidence that atmospheric transport is unimportant. Correlations between predictors can reduce their measured individual contributions under permutation. The results describe the predictive behavior of the retained M6 model rather than causal relationships.

Figure: `figures/fig_perm_importance.png`

### 3.2.2 Partial Dependence Analysis

Partial Dependence Plots were generated for wind speed and relative humidity using the M1 Baseline Decision Tree.

The baseline model was selected for this analysis because the M6 Ordinal Random Forest did not include the original `wind_speed_ms` and `relative_humidity_2m` variables among its 25 selected features.

Both PDP curves were relatively flat across the evaluated ranges, indicating limited variation in the baseline model's average predicted high-risk probability when these features were varied individually.

These plots describe model behavior rather than causal effects. Correlations between meteorological conditions and pollution features may limit the interpretation of partial dependence.

Figure: `figures/fig_pdp_wind_humidity.png`

## 3.3 Generalization Study

### 3.3.1 Spatial Generalization: Leave-One-Station-Out

A 26-fold Leave-One-Station-Out (LOSO) experiment was conducted using the M1 Decision Tree configuration. In each fold, one spatial node was excluded from training and used for evaluation. The dataset included three receptor nodes, ten corridor nodes, and thirteen source nodes.

Before model evaluation, the three-hour prediction targets were checked against available future observations. The QC-updated dataset contained 966,406 records. Of these, 966,071 satisfied the target-consistency verification criteria, while 335 records were excluded because their targets could not be verified.

The receptor-node results were as follows:

| Station | Evaluation Samples | Macro-F1 | High-Risk Recall | Average Asymmetric Cost |
|---|---:|---:|---:|---:|
| KL_City | 5,573 | 0.5652 | 0.6598 | 1.3277 |
| Port_Klang | 5,573 | 0.5946 | 0.5543 | 0.4263 |
| Sepang | 5,573 | 0.6024 | 0.6486 | 1.0074 |
| **Receptor mean** | — | **0.5874** | **0.6209** | **0.9205** |

The mean High-Risk Recall across the three receptor folds was 0.6209. The mean Average Asymmetric Cost was 0.9205. Performance differed by location, with Port_Klang producing the lowest asymmetric cost and KL_City the highest.

These findings indicate that the M1 baseline faces difficulties when making predictions at unseen spatial nodes. High-risk detection performance was not uniform across the three receptor locations.

The corridor and source nodes represent modeled background or transport-related grid locations. They should not be interpreted as 23 additional independent ground-monitoring stations.

The LOSO experiment reused previously engineered features. Consequently, it provides a spatial holdout assessment with partially precomputed inputs rather than a fully leakage-audited, end-to-end spatial validation.

Results: `models/loso_results.csv`

### 3.3.2 Temporal Generalization: 2025 to 2026

A temporal holdout experiment was conducted to assess the ability of the M1 Decision Tree to generalize to a later observation period. A fresh model was trained using receptor observations from 2025 and earlier and evaluated on receptor observations from 2026. No 2026 observations were used for model training.

Following the updated target-consistency checks, the experiment included 89,656 training observations and 21,852 evaluation observations across three receptor nodes.

The temporal holdout results were as follows:

| Metric | 2026 Evaluation |
|---|---:|
| Macro-F1 | 0.6740 |
| Balanced Accuracy | 0.6578 |
| High-Risk Recall | 0.7949 |
| MAOE | 0.3092 |
| Severe Jumping Error Rate | 0.0171 |
| Average Asymmetric Cost | 0.8178 |

The model achieved a High-Risk Recall of 0.7949, indicating that approximately 79.49% of high-risk observations were identified under the evaluation definition. The Average Asymmetric Cost was 0.8178, reflecting the operational consequences of classification errors under the project's cost matrix.

These results demonstrate that the M1 baseline retained measurable predictive performance when evaluated on observations from a later period. However, the temporal holdout results alone do not establish the magnitude of performance degradation, as this would require a comparable evaluation against an earlier period under the same experimental conditions.

The 2026 predictions were also used for the seasonal evaluation reported in Section 3.3.3.

### 3.3.3 Seasonal Generalization

Seasonal generalization was evaluated using the predictions from the 2026 temporal holdout experiment with the M1 Baseline Decision Tree. The evaluation observations were divided into two periods: June–October, defined as the project's monsoon-related period, and the remaining available months, January–May.

The same model, trained exclusively on receptor observations from 2025 and earlier, was evaluated for both periods. No seasonal retraining or additional threshold optimization was performed.

The QC-updated seasonal evaluation results are summarized below.

| Metric | June–October | January–May |
|---|---:|---:|
| Number of Samples | 10,998 | 10,854 |
| High-Risk Samples | 5,167 | 3,761 |
| Macro-F1 | 0.6812 | 0.6624 |
| Balanced Accuracy | 0.6645 | 0.6491 |
| High-Risk Recall | 0.8127 | 0.7705 |
| MAOE | 0.3187 | 0.2995 |
| Severe Jumping Error Rate | 0.0232 | 0.0110 |
| Average Asymmetric Cost | 0.8956 | 0.7390 |

The June–October period achieved a higher Macro-F1 and High-Risk Recall than January–May. High-Risk Recall increased by 4.22 percentage points, from 77.05% to 81.27%.

However, Average Asymmetric Cost was also higher during June–October, increasing from 0.7390 to 0.8956, a relative increase of approximately 21.19%. The Severe Jumping Error Rate was also higher during this period, at 0.0232 compared with 0.0110.

These results show that higher high-risk recall does not necessarily correspond to lower overall asymmetric prediction cost. Differences in class distributions and the severity of prediction errors may help explain the observed cost variation.

The seasonal comparison is descriptive rather than causal. The available 2026 observations end in October, meaning that the January–May group does not represent a complete November–May seasonal cycle. Further investigation of class distributions and error types would be required to explain the differences in model performance.

The seasonal evaluation results were saved in `models/seasonal_metrics.csv`.

## 3.4 Limitations

The experimental results should be interpreted in light of the following limitations:

1. **Validation-based threshold selection.** The M7 thresholds were selected exclusively using validation data to minimize Average Asymmetric Cost. Validation results are not independent test results. The fixed thresholds were subsequently evaluated on the test set without further optimization.

2. **Classification performance trade-offs.** Compared with the default decision rule, M7 increased test High-Risk Recall from 0.8737 to 0.9376 and reduced Average Asymmetric Cost from 0.6497 to 0.5405. However, Macro-F1 decreased from 0.6972 to 0.6659, and Balanced Accuracy decreased from 0.7192 to 0.6656. M7 therefore prioritizes reducing costly missed high-risk events rather than maximizing general classification performance.

3. **Reuse of the historical M6 model.** The M6 Ordinal Random Forest was not retrained after the upstream QC update. Its original fitted model and saved predictions were retained. Permutation importance used the compatible historical feature table from commit `67e8977`, with the model's original 25 feature names and ordering. Although predictions were successfully reproduced on aligned validation observations, this analysis does not measure the performance of a newly trained M6 model using the latest feature-selection configuration.

4. **Different models for interpretability analyses.** Permutation importance was calculated for M6, whereas the wind-speed and relative-humidity partial dependence plots were generated using the M1 Baseline Decision Tree. The PDP findings should not be interpreted as direct explanations of M6 or M7.

5. **Spatial generalization constraints.** The 26 spatial nodes included three receptor locations, ten corridor locations, and thirteen source locations. The corridor and source nodes represent modeled grid locations rather than independent ground-monitoring stations. The LOSO experiments also reused precomputed features, so the resulting estimates do not constitute a fully leakage-audited end-to-end spatial validation.

6. **Temporal and seasonal coverage.** The temporal experiment used historical receptor observations for training and observations from 2026 for evaluation. The available 2026 data ended in October. Consequently, the January–May seasonal group does not cover the full November–May period, and observed seasonal performance differences should not be interpreted as causal effects.

7. **Interpretability is not causality.** Permutation importance and partial dependence quantify aspects of model behavior. Correlations between predictors and interactions among pollution, weather, and spatial features limit their interpretation as independent physical effects.

These limitations are important when considering the reliability and deployment readiness of the proposed haze early-warning system.
