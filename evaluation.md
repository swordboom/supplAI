# Model Evaluation Report

Generated from: `models/model_sanity_report.json`  
Generated at (UTC): **2026-04-08T17:53:28.831918+00:00**

## Models Evaluated

1. Delay prediction model: XGBoost (CUDA) / RandomForest fallback (`models/delay_model.pkl`)
2. Anomaly detection model: IsolationForest (`models/anomaly_model.pkl`)

## Delay Model Metrics

| Metric | Train | Test |
|---|---:|---:|
| Accuracy | 0.9325 | 0.9192 |
| Precision | 0.8603 | 0.5774 |
| Recall | 0.2403 | 0.1647 |
| F1 | 0.3757 | 0.2563 |
| AUC | 0.9151 | 0.7936 |

### Overfitting Check

- AUC gap (train - test): **0.1216** -> **FAIL**
- Accuracy gap (train - test): **0.0133**

Interpretation:
- Model is learning signal, but there is **generalization drop** from train to test.

### Dataset Quality Signals (Delay Data)

- Rows used: **54977**
- Class distribution: **{'0': 0.91551, '1': 0.08449}**
- Exact duplicate ratio (feature+target): **0.4753** -> **FAIL**
- Top feature-target absolute correlations: **{'is_sunday_in_between': 0.132756, 'drop_metro': 0.093962, 'drop_non_metro': 0.093962, 'pickup_metro': 0.089621, 'pickup_non_metro': 0.089621, 'cp_ontime_per_quarter': 0.086058}**

## Bias/Fairness Indicators (Delay Model)

Indicators below are prediction-rate disparities across groups (higher means more imbalance):

- Pickup zone disparity: **0.0463**
- Drop zone disparity: **0.0343**
- SLA band disparity: **0.0000**

## Anomaly Model Metrics

- Feature rows (cities): **159**
- Duplicate city rows: **0.0000**
- Training anomaly rate: **0.0566**
- Graph anomaly rate: **0.0566**

### Anomaly Bias Indicators

- Country anomaly-rate disparity: **0.3333**
- Product anomaly-rate disparity: **0.4444**

### Injected Anomaly Recovery (Synthetic Stress Check)

- Known injected anomaly cities: **12**
- Detected injected anomaly cities: **7**
- Detection rate: **0.583333**

## Overall Health Summary

1. Training pipeline works and produces both model artifacts.
2. Inference pipeline is functional with pretrained artifacts.
3. Main quality risks to address next:
- Overfitting gap in delay model (AUC gap > 0.10)
- High duplicate ratio in delay dataset
- Group disparities in anomaly rates by country/product
