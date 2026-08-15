## Executive Summary

Loan defaults are costly and difficult to predict from surface-level applicant 
data alone — this project set out to test whether a data-driven, explainable 
model could meaningfully improve on that difficulty. Using the Home Credit 
Default Risk dataset (307,511 loan applicants), I built and evaluated a 
credit default prediction pipeline that combines applicant demographics, 
external credit bureau scores, and engineered features derived from 
applicants' prior loan history.

The final model (a tuned XGBoost classifier) achieves an AUC-ROC of 0.7687 
and a KS-statistic of 0.4046 on held-out test data — a meaningful, well-
validated improvement over a logistic regression baseline (AUC 0.7540). 
Beyond raw predictive performance, every prediction is paired with a SHAP-based 
explanation identifying exactly which factors drove that specific decision — 
a requirement for real-world credit risk deployment, where model decisions 
must be justifiable, not just accurate.

The model and its explanations are accessible via a live interactive 
dashboard: [https://github.com/jiggs108/credit-risk-dashboard]. This report documents the full pipeline: 
data cleaning, feature engineering, model development, evaluation, and 
explainability analysis.

## Problem Statement

Lenders face a persistent trade-off: approve too many risky applicants and 
losses mount from defaults; reject too many safe applicants and the business 
loses viable customers to competitors. Traditional credit scoring relies 
heavily on a small number of bureau-provided scores, but a lender's own 
application and prior-loan-history data often contains additional predictive 
signal that goes unused.

This project frames the problem as a binary classification task: given an 
applicant's demographic information, loan details, and history with prior 
credit products, predict the probability that they will default on the 
current loan (TARGET = 1) versus repay it successfully (TARGET = 0).

The dataset exhibits significant class imbalance — only 8.07% of applicants 
in the training data defaulted — which shaped both the modeling approach 
(favoring AUC-ROC and KS-statistic over raw accuracy) and the evaluation 
strategy (stratified train/test splitting) used throughout this project.

## Methodology

### 3.1 Data Overview

The primary dataset (`application_train.csv`) contains 307,511 loan 
applicants across 122 original features, including demographics, income, 
employment history, and loan details. Three supplementary datasets provided 
applicant-level history: `bureau.csv` (prior loans reported to credit 
bureaus), `installments_payments.csv` (payment-level history on prior 
installment loans), and `previous_application.csv` (prior applications with 
this lender).

Initial exploratory analysis found the three external credit bureau scores 
(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) to be the strongest available 
predictors of default (correlations of -0.155 to -0.179 with the target), 
notably stronger than commonly-assumed predictors like raw income, which 
showed almost no separation between defaulters and non-defaulters in 
exploratory analysis.

### 3.2 Data Cleaning

Several data quality issues required resolution before modeling:

- **Placeholder values**: DAYS_EMPLOYED contained a placeholder value 
  (365243, equivalent to ~1,000 years) in 55,374 rows, corrected to null.
- **High-missingness columns**: 40 columns with >50% missing data were 
  dropped, reducing the feature set from 122 to 82 columns — with one 
  exception: EXT_SOURCE_1 was retained despite ~56% missingness due to its 
  outsized predictive value, illustrating that missingness thresholds 
  should defer to known feature importance rather than being applied 
  mechanically.
- **Remaining missing values**: Median imputation was applied to numeric 
  fields with a small number of missing values, and zero-imputation to 
  count-based fields (e.g., credit bureau inquiry counts) where a missing 
  value plausibly represented "no data available" rather than "unknown."
- **A gap in the original cleaning pass**: 18 columns (building/property 
  characteristics, ~48-50% missing — just under the drop threshold) were 
  inadvertently left unimputed, surfacing as ~1 million null values during 
  Week 3 model training. This was identified and corrected before modeling 
  proceeded, using the same missingness-pattern reasoning applied elsewhere 
  in the pipeline.

  ### 3.3 Feature Engineering

Beyond the application-level data, three supplementary datasets provided an 
opportunity to capture applicants' prior credit behavior — information not 
present in the base application table.

**Bureau data (`bureau.csv`)**: An initial feature counting each applicant's 
total number of prior loans (PREV_LOAN_COUNT) showed essentially no 
correlation with default (-0.01), indicating that loan *quantity* alone 
carries little signal. Follow-up features measuring loan *quality* — 
particularly ACTIVE_LOAN_RATIO (proportion of prior loans still active) — 
performed better (0.049), and restricting overdue-payment features to the 
most recent 2 years roughly doubled their correlation strength (e.g., 
AVG_DAYS_OVERDUE: 0.007 → RECENT_AVG_DAYS_OVERDUE: 0.016), confirming that 
recent behavior is more predictive than lifetime history.

**Installment payment history (`installments_payments.csv`)**: Row-level 
payment data (13.6 million rows) was aggregated to applicant level. 
LATE_PAYMENT_COUNT — the frequency of late payments — proved more predictive 
(0.032) than the severity of any single late payment (MAX_PAYMENT_DELAY: 
0.005), suggesting that consistency of payment behavior matters more than 
one-off delays.

**Prior application history (`previous_application.csv`)**: REFUSAL_RATE — 
the proportion of an applicant's previous loan applications that were 
refused — emerged as the strongest engineered feature in the project 
(correlation 0.078), approaching the strength of raw application-level 
predictors and later confirmed as a top-10 driver in the final model's 
SHAP-based feature importance (see Section 5).

In total, 14 engineered features were constructed and carried into modeling, 
alongside the cleaned application-level features. A full feature dictionary 
is provided in Appendix A.

### 3.4 Modeling Approach

Categorical variables were one-hot encoded (13 columns, expanding the 
dataset from 96 to 196 columns), and the data was split into training and 
test sets (80/20) using stratified sampling to preserve the dataset's 
8.07% default rate in both sets.

Three modeling approaches were evaluated in sequence, each building on the 
last:

1. **Logistic regression** (baseline) — a simple, interpretable linear 
   model, included deliberately as a floor for comparison rather than a 
   candidate for deployment.
2. **XGBoost (default parameters)** — a gradient-boosted tree ensemble, 
   selected for its strong track record on structured/tabular data and its 
   ability to capture non-linear feature interactions without manual 
   feature transformation.
3. **XGBoost (tuned)** — hyperparameters optimized via RandomizedSearchCV 
   (20 iterations, 3-fold cross-validation, optimizing for AUC-ROC), 
   chosen over an exhaustive grid search for practicality given dataset 
   size and compute constraints.

Models were evaluated primarily on AUC-ROC and the KS-statistic rather than 
accuracy, given the dataset's significant class imbalance (8.07% default 
rate) — a naive model predicting "no default" for every applicant would 
achieve ~92% accuracy while providing no practical value to a lender. Both 
AUC-ROC and KS-statistic evaluate a model's ability to *rank* applicants by 
risk across all possible decision thresholds, which better reflects how 
such a model would actually be used in practice: to prioritize and 
differentiate risk, with the final approval threshold set as a separate 
business decision.

## 4. Results

Three models were evaluated on held-out test data (61,503 applicants, 
never used during training or hyperparameter tuning):

| Model | AUC-ROC | KS-Statistic | Train-Test AUC Gap |
|---|---|---|---|
| Logistic Regression (baseline) | 0.7540 | 0.3819 | — |
| XGBoost (default parameters) | 0.7645 | 0.3959 | 0.0298 |
| XGBoost (tuned) | **0.7687** | **0.4046** | 0.0374 |

![ROC Curve Comparison](final_roc_comparison.png)

The tuned XGBoost model was selected as the final model. While the 
improvement over the untuned XGBoost baseline was modest (+0.42% AUC), the 
KS-statistic crossing the 0.4 threshold represents a meaningful practical 
improvement by credit-scoring industry standards, where a KS above 0.4 is 
generally considered strong separation between risk classes. The 
progression from baseline to tuned model was consistent and incremental 
across both metrics — a pattern more indicative of genuine, well-validated 
signal than of overfitting or leakage, which would typically produce 
larger, less consistent jumps.

**Hyperparameter search**: RandomizedSearchCV (20 iterations, 3-fold 
cross-validation) identified the following optimal configuration: 
`n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, 
colsample_bytree=0.8`. Notably, the optimal max_depth matched the 
initial default-parameter guess, suggesting the original untuned model was 
already reasonably well-configured for this dataset — the primary gains 
from tuning came from a lower learning rate paired with more estimators 
(a more cautious, gradual learning process) combined with subsampling for 
regularization.

**Threshold selection**: Rather than using the conventional 0.5 
classification threshold, the KS-optimal threshold (0.0846) was identified 
and used for classification throughout this project. This threshold sits 
close to the dataset's actual base default rate (8.07%), and reflects an 
important lesson from early baseline testing: at a 0.5 threshold, the 
logistic regression baseline classified only 150 of 4,965 actual defaulters 
as high-risk — not because the model's underlying risk ranking was poor, 
but because 0.5 is not a meaningful decision boundary when the true event 
rate is only ~8%.

## 5. Model Explainability

A model's predictive performance alone is insufficient for real-world 
credit risk deployment — regulators and business stakeholders typically 
require the ability to explain *why* a specific applicant received a 
specific risk score. SHAP (SHapley Additive exPlanations) was used to 
provide this at both a global (whole-model) and local (individual 
prediction) level.

**Global feature importance**: Across a representative sample of the test 
set, the three external credit bureau scores (EXT_SOURCE_1, EXT_SOURCE_2, 
EXT_SOURCE_3) were confirmed as the dominant drivers of model predictions — 
consistent with their strength in the original correlation analysis (see 
Section 3.1). Notably, two engineered features from Section 3.3 — 
AVG_PAYMENT_SHORTFALL and REFUSAL_RATE — ranked within the top 10 features 
by importance, confirming that the feature engineering work contributed 
genuine, model-level signal rather than only marginal statistical 
correlation.

**Directional validation**: SHAP summary plots confirmed that low 
EXT_SOURCE values and high REFUSAL_RATE/AVG_PAYMENT_SHORTFALL values 
consistently pushed predictions toward higher default risk, and vice versa — 
a sensible, explainable relationship that held at the level of individual 
predictions, not just in aggregate.

**Individual prediction analysis**: SHAP force plots were used to examine 
specific applicant predictions in detail. One case of particular note was a 
false positive: an applicant predicted at 63.1% default probability who did 
not, in fact, default. Analysis of this prediction showed the model's high-
risk assessment was driven by genuinely weak signals — low EXT_SOURCE_2 and 
EXT_SOURCE_3 scores combined with an 80% prior loan refusal rate — meaning 
the prediction was well-reasoned and defensible given the information 
available, even though the outcome ultimately differed. This distinction — 
between a model being *wrong* and a model being *unreasonable* — is central 
to deploying explainable ML in a regulated, high-stakes context like credit 
risk.

An interactive version of this explainability analysis, allowing exploration 
of custom applicant profiles, is available in the accompanying dashboard: 
[https://github.com/jiggs108/credit-risk-dashboard].

## 6. Limitations & Future Work

**Feature ceiling**: The strongest available predictors (EXT_SOURCE_1/2/3) 
are external, pre-computed scores whose underlying methodology is not 
disclosed. Even the best-performing engineered feature (REFUSAL_RATE, 
0.078 correlation) fell well short of these external scores, suggesting 
the practical ceiling for this dataset's predictive power may be more 
constrained by available feature richness than by modeling technique — 
a conclusion supported by the modest, incremental gains observed across 
all three modeling approaches.

**High-cardinality categorical encoding**: ORGANIZATION_TYPE (58 unique 
values) and OCCUPATION_TYPE (18 unique values) were one-hot encoded 
directly, contributing over half of the total column expansion from 
encoding. A future iteration could group these into broader categories or 
use target encoding to reduce dimensionality without losing signal.

**Demographic features in model importance**: CODE_GENDER_M appeared 
within the top 10 features in the SHAP global importance analysis. While 
this reflects a genuine statistical pattern in the training data, any 
real-world deployment of this model would require careful fairness and 
regulatory compliance review before using demographic attributes as model 
inputs — a step outside the scope of this exploratory project but essential 
in practice.

**Sample-based SHAP analysis**: For computational efficiency, SHAP values 
were computed on a representative sample (1,000 rows) of the test set 
rather than the full test set. Given the sample size and the consistency of 
results with prior correlation analysis, this is unlikely to materially 
affect conclusions, but a production deployment would benefit from 
full-dataset SHAP analysis.

**Static train/test split**: Model evaluation used a single stratified 
train/test split rather than repeated cross-validation on the final held-out 
metrics. Given the dataset's size (307,511 rows), this is a reasonable 
trade-off for a project of this scope, but a production model would benefit 
from more rigorous validation (e.g., temporal validation, if application 
date data were available, to simulate real deployment conditions more 
closely).

## 7. Conclusion

This project demonstrates an end-to-end credit risk modeling pipeline — 
from raw, messy multi-table data through feature engineering, model 
selection, and explainability analysis — culminating in a tuned XGBoost 
classifier (AUC-ROC 0.7687, KS-statistic 0.4046) with fully explainable, 
individual-level predictions via SHAP.

Beyond the modeling result itself, this project reflects an approach to 
data science grounded in explicit reasoning: every major decision — which 
columns to drop, which imputation strategy to use, which evaluation metric 
to optimize, which threshold to classify at — was made deliberately and 
documented, rather than defaulted to convention. This is the standard the 
project was held to throughout, and the standard I'd bring to real-world 
data science and AI engineering work.

The full pipeline, model, and interactive dashboard are available in this 
repository, with the live application accessible at: [https://github.com/jiggs108/credit-risk-dashboard].

