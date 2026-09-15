# Credit Default Prediction (ML Project)

This is a follow-up to my [SQL Credit Risk Analysis Project](https://github.com/Prerit-srivastava12/credit-risk-analysis-project) using the same two datasets (Lending Club and L&T Financial Services), but this time building actual predictive models instead of descriptive SQL analysis. The SQL project answered "what patterns exist in past defaults"; this one asks "can we predict default before it happens, and which factors matter most."

Same two-module structure as before: Lending Club (US, full application-time feature set) and L&T NBFC (India, underwriting risk). Each module builds a logistic regression baseline and an XGBoost model, compares them, and uses SHAP to explain what the best model actually learned.

## Stack
- PostgreSQL (same database from the SQL project)
- Python: pandas, scikit-learn, XGBoost, SHAP
- Jupyter notebooks in VS Code

## Running this yourself
1. Have the SQL project's database [credit_risk_analysis_project](https://github.com/Prerit-srivastava12/credit-risk-analysis-project) set up and loaded as this project reads directly from it.
2. Set your own PostgreSQL password in the connection string before running, both notebooks use a placeholder (`YOUR_PASSWORD_HERE`) instead of real credentials.
3. Open [credit_default_model_lending_club.ipynb](credit_default_model_lending_club.ipynb) and [credit_default_model_lt_nbfc.ipynb](credit_default_model_lt_nbfc.ipynb) and run the cells top to bottom.

## Repo structure
```
credit_default_model_lending_club.ipynb  : Module 1: Lending Club
credit_default_model_lt_nbfc.ipynb       : Module 2: L&T NBFC
```

---

## Module 1: Lending Club

### The feature question: what's actually usable at application time?
A model can only use information that exists *before* a loan is approved, not information that only exists once we already know the outcome. `loan_status` is the target, not a feature. `interest_rate` and `installment` are set *after* Lending Club assigns a grade, so they leak grade-derived information. `grade` and `sub_grade` themselves are a genuine gray area.In simple terms they're Lending Club's own internal risk score, assigned at approval time using information we don't have. Rather than picking a side, I built two versions: **Version A** uses only raw application-time features (income, DTI, purpose, employment, home ownership, loan amount, term); **Version B** adds grade/sub_grade back in, to see how much predictive power comes from Lending Club's own proprietary score versus what's recoverable from raw data alone.

### Class imbalance problem
The baseline logistic regression (Version A) hit 80% accuracy but only caught 1% of actual defaults, the model had essentially learned to always predict "safe," since defaults are only ~20% of the data and accuracy doesn't punish that. Using `class_weight='balanced'` fixed this: recall on defaults jumped to 57%, at the cost of accuracy dropping to 64%. ROC-AUC barely moved (0.660 to 0.660 give or take), which makes sense once you know what it measures, it's about ranking ability, not where the decision cutoff sits, so changing the threshold weighting doesn't change it much on its own.

### Full model comparison

| Model | Features | ROC-AUC | Recall (default) |
|---|---|---|---|
| Logistic Regression | No grade | 0.660 | 0.57 |
| Logistic Regression | With grade | 0.706 | 0.68 |
| XGBoost | No grade | 0.667 | 0.59 |
| **XGBoost** | **With grade** | **0.709** | **0.67** |

XGBoost only marginally beat logistic regression on the same features (0.667 vs 0.660) which is a much smaller jump than adding grade gave either model (roughly +0.04-0.05 either way). The real lesson here: **feature choice mattered more than model choice.** Grade clearly carries real signal beyond what raw application data captures, but the gap isn't huge — an independently-built model on raw origination data alone gets you most of the way to Lending Club's own proprietary score.

### One bug worth mentioning
XGBoost rejected a one-hot encoded column name containing `<` (from the `employment_years` category `< 1 year` — XGBoost doesn't allow `<`, `[`, or `]` in feature names). Fixed by relabeling that category to `less than 1 year` in the source data before encoding, rather than patching column names after the fact.

### SHAP findings (best model: XGBoost with grade)
Term and grade dominate. 60-month loans and worse letter grades both push predictions toward higher default risk, and the grade effect climbs smoothly and correctly from B through G. Beyond that: higher debt-to-income, lower income, and larger loan amounts all push toward higher predicted risk, all in the direction you'd expect. Mortgage-holders came out predicted as comparatively safer than renters — plausible, since holding a mortgage generally signals more financial stability.

---

## Module 2: L&T NBFC

Fewer usable features here than Lending Club : no purpose, no income, no internal grade equivalent. Just disbursed amount, asset cost, LTV, bureau score, and employment type. This alone turns out to explain a lot of the performance gap between the two modules.

### The bureau_score = 0 problem
Half the dataset (116,950 of 233,154 rows) had a bureau score of exactly 0 which is not a real low score, but a placeholder for "no credit history on file." Treating it as a real score would have badly distorted the model's sense of what a low score means. Fixed by splitting this into two pieces: a `has_credit_history` flag (1/0), and a `bureau_score_clean` column where the placeholder zeros become genuine missing values instead of a fake number.

This meant logistic regression (which can't handle missing values at all) had to train on only the 116,950 borrowers with a real score, while XGBoost which handles missing values natively could use the full dataset, including the credit-invisible borrowers. That's a genuine practical advantage of tree-based models for real-world lending data, and it's directly relevant here: a large share of India's population is new-to-credit or has no bureau history, so a model that can only work with scored borrowers is missing a real, common segment.

### Model comparison

| Model | Data used | ROC-AUC | Recall (default) |
|---|---|---|---|
| Logistic Regression | ~116K, scored borrowers only | 0.594 | 0.58 |
| **XGBoost** | ~233K, full dataset | **0.612** | **0.62** |

Both models are meaningfully weaker than Lending Club's which lines up with having a much thinner feature set to work with. XGBoost's edge here likely comes mostly from being able to use the full dataset (including unscored borrowers), not just from being a fundamentally stronger algorithm.

### SHAP findings
Loan-to-value ratio came out as the single strongest predictor, and usefully, in exactly the direction theory predicts: higher LTV pushes toward higher risk. This is worth noting against the SQL project's finding, where the LTV bucket analysis showed an odd Mid > High reversal, which I attributed to the High-LTV bucket having a tiny sample (1,219 loans). SHAP, working off the full dataset rather than three coarse buckets, resolves that inconsistency and shows the expected relationship clearly.

Bureau score and employment type also behaved sensibly, self-employed borrowers and borrowers with unknown employment type both skewed riskier. The one surprise: `has_credit_history` barely mattered to the model at all, despite covering half the dataset. Best explanation: XGBoost's native handling of the missing bureau score already captures "no credit history" as an effective signal on its own, making the explicit flag largely redundant.

---

## What connects the two modules
Same conclusion showed up independently in both: **feature richness mattered more than model sophistication.** In Lending Club, adding one feature (grade) moved ROC-AUC more than switching from logistic regression to XGBoost did. In L&T, the model with a thinner feature set never got close to Lending Club's numbers, no matter which algorithm was used. XGBoost's one consistent, genuine edge over logistic regression in both modules was its ability to natively handle missing/imbalanced data — not needing to drop rows or manufacture placeholder values the way logistic regression required.

## Limitations
- Lending Club's Version A/B split is a judgment call about data leakage, not a hard rule, reasonable people could include grade differently.
- The L&T `has_credit_history` flag added little value on top of XGBoost's native missing-value handling, worth dropping in a future iteration rather than keeping as dead weight.
- Neither module's ROC-AUC (0.6-0.71) reaches the level of a production-grade credit model, these are honest, real-world-messy first models, not final systems.
- Both modules apply `class_weight`/`scale_pos_weight` to handle imbalance rather than more advanced techniques (e.g. SMOTE), which could be a natural next step.
