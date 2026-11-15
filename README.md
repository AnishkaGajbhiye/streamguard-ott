# StreamGuard OTT: Customer Churn Prediction

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnishkaGajbhiye/streamguard-ott/blob/main/Streamguard_OTT_Model.ipynb)

A machine learning notebook that predicts which streaming-service customers will cancel, then uses those predictions to estimate how much monthly revenue is at risk.

> **Important:** the data is a synthetic Netflix-style dataset from Kaggle (5,000 customers). The scores below show what the models can learn from *this* data. They are not how the models would perform on real customers. This is a notebook pipeline, not a deployed app.

## Results

| Test set (1,000 held-out customers) | F1 | ROC-AUC | False alarms | Missed churners |
|---|---|---|---|---|
| Tuned AdaBoost, all features | 0.990 | 0.999 | 6 | 4 |
| Tuned AdaBoost, **without** `last_login_days` | 0.764 | 0.874 | 120 | 118 |

**Key finding:** most of the model's accuracy comes from one column, `last_login_days`. A customer who has already cancelled would naturally not have logged in recently, so this column may partly reflect the outcome instead of predicting it. I report both results because it isn't clear whether that value would be known *before* a customer cancels.

## What the notebook does

1. **Data checks:** shape, missing values, duplicates, outliers and category balance. `avg_watch_time_per_day` turned out to be just `watch_hours / (last_login_days + 1)`, so it was dropped as redundant.
2. **Leak-safe preprocessing:** stratified 80/20 split first; the outlier cap on `watch_hours` is computed from training data only; imputation, scaling and one-hot encoding live in a scikit-learn `Pipeline` fitted on training data only.
3. **Feature engineering:** `engagement_index` (hours watched per dollar of monthly fee).
4. **Model comparison** with stratified 5-fold cross-validation:

   | Model | CV F1 | CV ROC-AUC |
   |---|---|---|
   | Logistic Regression | 0.892 | 0.964 |
   | KNN (k=5) | 0.864 | 0.933 |
   | SVM (RBF) | 0.916 | 0.976 |
   | Random Forest | 0.968 | 0.995 |
   | AdaBoost | 0.970 | 0.996 |

5. **Tuning:** `RandomizedSearchCV` (15 settings, 5-fold) for Random Forest and AdaBoost. AdaBoost was picked from cross-validation results alone (best CV F1 0.987), before the test set was touched.
6. **Checks on the result:** feature selection (RFE), churn rate by group, and ablations that remove columns one at a time. These are what exposed the `last_login_days` effect.
7. **Business layer:**
   - *Revenue at risk* = churn probability × monthly fee. The model expects about $6,638 per month at risk, against $6,444 in fees actually paid by customers who churned. Premium has the most revenue at risk ($2,673) even though Basic churns the most, because Premium customers pay more.
   - *12-month customer value* = monthly fee × 12 × (1 − churn probability), with Low / Medium / High risk tiers. Actual churn in each tier was 0.0%, 30.2% and 99.7%, so the tiers rank customers in the right order.

## Limitations

- The data is synthetic, so the near-perfect score should not be read as real-world performance.
- No tenure or support-ticket columns are available.
- Model probabilities are not calibrated, so they are better for *ranking* customers than as exact chances.
- The 12-month value and the risk-tier cut-offs (0.3 / 0.6) are my own assumptions.
- There is no evidence here that any retention offer works; that would need an experiment.

## Next steps

- Build a simple interface (for example a Streamlit dashboard) to browse customers by risk tier.
- Test on real customer data that includes tenure and support information.
- Run an experiment to see whether retention offers actually keep customers.

## How to run

**Easiest:** click the "Open in Colab" badge above and choose *Runtime → Run all*.

**Locally:**

```bash
git clone https://github.com/AnishkaGajbhiye/streamguard-ott.git
cd streamguard-ott
pip install -r requirements.txt
jupyter notebook Streamguard_OTT_Model.ipynb
```

The notebook downloads the dataset automatically with `kagglehub`.

## Data

[Netflix Customer Churn Dataset](https://www.kaggle.com/datasets/abdulwadood11220/netflix-customer-churn-dataset) by abdulwadood11220 on Kaggle (synthetic, 5,000 rows, 14 columns). The data is not stored in this repository.

## Tech

Python, pandas, NumPy, scikit-learn, SciPy, kagglehub, Jupyter / Google Colab.

## Author

Anishka Gajbhiye, second-year Computer Engineering student at SPIT, Mumbai.
[LinkedIn](https://www.linkedin.com/in/anishka-gajbhiye-181593395/)
