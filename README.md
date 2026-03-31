# Sleep Disorder Classification

Classifying sleep disorders (None, Insomnia, Sleep Apnea) from physiological and
lifestyle data using logistic regression and random forest.

## Background
Poor sleep is linked to cardiovascular disease, cognitive decline, and metabolic
disorders — but most people don't know they have a sleep disorder until symptoms
are severe. Early detection from lifestyle and biometric data is a tractable ML
problem and a natural extension of my previous work in fitness analytics.

This is my second data science project. The first was a workout analytics project
using pandas and matplotlib. This one focuses on building a complete supervised ML
workflow — data cleaning, feature engineering, model training, and honest evaluation.

## Dataset
- **Source:** [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) — Kaggle
- **Size:** 374 rows, 13 original features
- **Target:** Sleep Disorder — None, Insomnia, or Sleep Apnea
- **Caveat:** The dataset is synthetic. Performance metrics reflect a working ML
  pipeline, not clinically valid predictions. This model should not be interpreted
  as generalizable to real patient data.

## Project Structure
```
sleep-health-ml/
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv
├── notebooks/
│   │── 01_eda.ipynb
│   └── 02_model.ipynb
├── src/
│   └── __init__.py
├── README.md
└── .gitignore
```

## Methods

**Cleaning**
- Replaced NaN values in Sleep Disorder with "None" (absence of disorder, not missing data)
- Consolidated "Normal" and "Normal Weight" BMI labels into a single category
- Grouped low-frequency occupations (≤4 rows) into "Other"

**Feature Engineering**
- Split Blood Pressure from a single string column into BP Systolic and BP Diastolic
- Engineered Pulse Pressure (Systolic − Diastolic) and dropped the two source columns
- Label encoded BMI Category (ordinal) and Sleep Disorder (target)
- One-hot encoded Gender and Occupation

**Modeling**
- Established a majority-class dummy classifier as a performance baseline
- Trained logistic regression and random forest classifiers
- Scaled features with StandardScaler prior to logistic regression
- Stratified train/test split (80/20) to preserve class distribution

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Baseline (Most Frequent) | 0.587 | 0.246 | 0.434 |
| Logistic Regression | 0.893 | 0.866 | 0.899 |
| Random Forest | 0.920 | 0.889 | 0.922 |

Random forest was the best performing model, achieving 0.92 accuracy and 0.889 macro 
F1 — a 0.643 point improvement over the dummy baseline on macro F1.

**Top predictive features (Random Forest):**
- BMI Category (0.21) — strongest single predictor
- Age (0.15)
- Sleep Duration (0.13)
- Occupation_Nurse (0.12) — likely a gender confound in this dataset rather than 
  a genuine occupational effect
- Heart Rate (0.07)

The model cleanly separates disorder cases from non-disorder cases. The primary 
confusion is between Insomnia and Sleep Apnea — both models misclassified 3 Insomnia 
cases as Sleep Apnea, suggesting physiological overlap between the two disorders in 
this dataset.

## Limitations

- **Synthetic data:** The dataset was algorithmically generated, not collected from 
  real patients. Distributions are artificially clean and do not reflect real-world 
  variability. Model performance would likely degrade significantly on real clinical data.

- **Small sample size:** 374 rows is insufficient for a clinically meaningful model. 
  Class-level evaluation (15–16 test cases per disorder) produces unstable metrics 
  that would shift with a different random seed.

- **Nurse/Sleep Apnea confound:** Occupation_Nurse ranked as the 4th most important 
  feature, but this is likely a gender effect — nurses in this dataset are 
  predominantly female, and females have a disproportionately high Sleep Apnea rate. 
  The occupational signal is probably not real.

- **Correlated features:** Sleep Duration, Quality of Sleep, and Stress Level are 
  highly intercorrelated (r = 0.88, -0.90). Feature importance scores for these 
  variables should be interpreted cautiously — importance is split arbitrarily 
  between correlated features in tree-based models.

- **No hyperparameter tuning:** Models were trained with default parameters. 
  Performance could improve with cross-validation and tuning, but given the dataset 
  size this risks overfitting to a small test set.

  ## Next Steps

- Replicate the analysis on a real dataset — NHANES or a personal wearables export 
  (Apple Health, Fitbit) — to test whether these patterns hold on non-synthetic data
- Add cross-validation to produce more stable performance estimates given the small 
  sample size
- Investigate the gender/occupation confound by stratifying models by gender and 
  comparing feature importance
- Explore additional engineered features from Blood Pressure (mean arterial pressure, 
  hypertension classification)
- Address class imbalance more explicitly using SMOTE or class-weighted models

## Tools
- **Python 3.12.5** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Jupyter Notebooks** — exploratory analysis and modeling
- **VS Code** — development environment
- **Git / GitHub** — version control and portfolio hosting