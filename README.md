# Week 4 Capstone — Predicting Internship Selection Outcomes

A beginner-friendly, reproducible machine-learning capstone exploring internship selection outcomes using a small, partly simulated dataset.

## Objective
Combine data cleaning, exploratory analysis and aggregation skills from Weeks 1–3 with a simple binary classification model, transparent evaluation and clear communication for a non-technical audience.

## Dataset description
- Source: Week 1 cleaned applicant dataset (`data/applicants_cleaned.csv`), 73 rows, 7 original columns.
- Outcome: 23 Selected, 16 Rejected, 34 Under Review. The 39 confirmed outcomes are used for modeling; Under Review records are excluded.
- `Domain Applied` has missing values; confirmed records with missing domains are assigned an `Unknown` category before one-hot encoding.
- Four simulated, not historically observed features: number of skills (1–8), portfolio (Yes/No), prior hackathon participation (Yes/No), statement quality (1–10). Reproducible random seed 42.
- Applicant names, email addresses and phone numbers are excluded from model features. The supplied dataset contains personal identifiers; do not publish the raw CSV publicly without appropriate permission and de-identification. The repository can omit the raw CSV and provide a locally loaded authorized copy instead.

## Method
1. Load and inspect the cleaned Week 1 data; summarize missing values and selection status.
2. Add four simulated features with `np.random.seed(42)` and explain their limitations.
3. Convert Yes/No to 1/0, create a Selected=1/Rejected=0 target, exclude Under Review, one-hot encode application domains (`drop_first=True`).
4. Split the confirmed outcomes into 80% training / 20% testing, with `random_state=42` and stratification.
5. Train Logistic Regression (`max_iter=1000`) and predict test outcomes.
6. Report accuracy, precision, recall, confusion matrix and coefficient interpretation.

## Key results
- Training applicants: 31; held-out test applicants: 8.
- Accuracy: 37.50% (3 / 8 correct).
- Precision: 50.00%.
- Recall: 40.00%.
- Confusion matrix (`[[TN, FP], [FN, TP]]`): **[[1, 2], [3, 2]]**.

These metrics are illustrative only: the test set is extremely small and four inputs are randomly simulated. The model must not be used for actual selection decisions. Historical selection decisions may also contain bias. Human review and larger, genuine datasets are essential.

## How to run
1. Install Python 3 and dependencies: `pip install -r requirements.txt`.
2. Open `Week4_Capstone_Jamsad_Chowdhury.ipynb` in Jupyter or Google Colab.
3. Make sure the authorized `applicants_cleaned.csv` is available in `data/` (or the notebook's working directory). If using Colab, upload the file and adjust the path as needed.
4. Run all notebook cells **top to bottom**. The notebook saves figures and a coefficient table in `outputs/`.

## Recorded walkthrough (2–3 minutes)
Video link: https://drive.google.com/file/d/16JEkWrE47tfKAS1gBDgZBkRt3ADPvDk9/view?usp=sharing

## Recommendation
Collect genuine skills, portfolio, hackathon and statement data consistently, then validate the approach on a larger set of confirmed outcomes before considering any real-world use.
