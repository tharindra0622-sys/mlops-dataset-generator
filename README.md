# MLOps Dataset Generator
## For Research: Early Prediction of CI/CD and MLOps Pipeline Failures

---

## What this generates

A labelled MLOps pipeline failure dataset with these features per run:

| Category          | Features |
|-------------------|----------|
| Pipeline metadata | trigger, OS, duration, queue wait |
| Data features     | size, missing %, drift, schema errors |
| Model features    | algorithm, hyperparameters, training time |
| Evaluation        | metric score, threshold pass/fail, overfitting |
| Error signals     | OOM, schema, convergence, dependency errors |
| **Target**        | `conclusion` = success / failure, `label` = 0/1 |

### Failure scenarios injected

| Scenario | Type | Description |
|----------|------|-------------|
| `missing_data_file` | Data | Dataset file not found |
| `schema_mismatch` | Data | Wrong column names |
| `memory_overflow` | Data | OOM error on large arrays |
| `wrong_hyperparameters` | Model | Invalid learning rate/estimators |
| `eval_threshold_fail` | Eval | Metric below required threshold |
| `dependency_conflict` | Env | Package version incompatibility |
| `drift + missing` | Data | Statistical drift + missing values |

---

## Setup — Step by Step

### Step 1 — Fork / create the repo on GitHub

```bash
# Create a new GitHub repo named: mlops-dataset-generator
# Then clone it locally
git clone https://github.com/YOUR_USERNAME/mlops-dataset-generator
cd mlops-dataset-generator
```

### Step 2 — Copy these files into the repo

```
.github/
  workflows/
    generate_dataset.yml     ← GitHub Actions workflow
scripts/
  train_pipeline.py          ← Main MLOps pipeline with failure injection
  collect_dataset.py         ← Builds master CSV from JSON logs
  download_and_build.py      ← Downloads artifacts from GitHub
requirements.txt
README.md
```

### Step 3 — Push to GitHub

```bash
git add .
git commit -m "feat: add MLOps dataset generator pipeline"
git push origin main
```

### Step 4 — Enable GitHub Actions

- Go to your repo on GitHub
- Click **Actions** tab
- Click **"I understand my workflows, go ahead and enable them"**

### Step 5 — Trigger the first run manually

- Go to **Actions → MLOps Dataset Generator**
- Click **"Run workflow"**
- Leave defaults and click **"Run workflow"**

Each workflow run generates approximately **40–60 individual pipeline runs**
across all job matrices (success + failure scenarios).

### Step 6 — Let it accumulate (recommended: 2–4 weeks)

The workflow runs automatically every 2 hours via the cron schedule.

| Duration | Approximate runs generated |
|----------|---------------------------|
| 1 week   | ~500–700 runs |
| 2 weeks  | ~1,000–1,400 runs |
| 1 month  | ~2,000–3,000 runs |

### Step 7 — Download and build the dataset locally

```bash
# Install dependencies
pip install requests pandas scikit-learn mlflow

# Set your GitHub Personal Access Token
# (Settings → Developer settings → Personal access tokens → Fine-grained)
# Required permissions: Actions (read), Contents (read)
export GITHUB_TOKEN=ghp_your_token_here

# Download all artifacts and build master CSV
python scripts/download_and_build.py --repo YOUR_USERNAME/mlops-dataset-generator

# Build the final dataset CSV
python scripts/collect_dataset.py
```

### Step 8 — Use the dataset

The final output is at `data/mlops_dataset.csv` with:
- All feature columns
- `conclusion` column: success / failure
- `label` column: 0 (success) / 1 (failure)

```python
import pandas as pd
df = pd.read_csv('data/mlops_dataset.csv')
print(df['label'].value_counts())
print(df.shape)
```

---

## Expected class distribution

| Conclusion | Approx % |
|------------|----------|
| success    | 55–65%   |
| failure    | 35–45%   |

This is better balanced than the CI/CD dataset (~70/30) because
failure scenarios are explicitly injected.

---

## Combining with your CI/CD dataset

```python
import pandas as pd

# Load both datasets
df_cicd  = pd.read_csv('runs_final.csv.zip')
df_mlops = pd.read_csv('data/mlops_dataset.csv')

# Add source label
df_cicd['pipeline_category']  = 'cicd'
df_mlops['pipeline_category'] = 'mlops'

# Common columns only
common_cols = ['duration_sec', 'label', 'pipeline_category',
               'runner_os', 'conclusion', 'trigger_event']
df_combined = pd.concat([
    df_cicd[common_cols],
    df_mlops[common_cols]
], ignore_index=True)

print(df_combined['pipeline_category'].value_counts())
print(df_combined['label'].value_counts())
```

---

## Citing this dataset

If you use this in your thesis or paper:

> Tharindra et al. (2025). "MLOps Pipeline Failure Dataset Generated via
> GitHub Actions Failure Injection." University of Peradeniya.
> Available at: https://github.com/YOUR_USERNAME/mlops-dataset-generator

---

## Estimated GitHub Actions usage

Free tier: **2,000 minutes/month**

Each workflow run uses approximately **15–25 minutes** total.
At every-2-hours schedule: ~360 scheduled triggers/month.
With matrix jobs, actual compute is parallelised.

**Recommendation:** Set the cron to `0 */6 * * *` (every 6 hours)
to stay within the free tier comfortably.
