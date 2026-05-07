# Skill Job Role Predictor - ML

This project trains a supervised machine learning model that predicts a job role from job-related features such as skills, experience level, and education level.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the reproducible training pipeline:

```bash
python src/train_model.py
```

The script resolves paths relative to its own file location, so the model artifacts are saved consistently inside this project.

For parser scripts only, install Playwright browsers:

```bash
playwright install
```

## Problem & Goal
At now graduated students know their stack but not sure about best job role for them this project will 
solve this problem by suggesting best job roles for them based on their skills education and experience
## Dataset Collection & Augmentation
The dataset was collected and merged from headhunter.kz and qyzmet.kz using Playwright/Selenium scripts.
- **Data Augmentation:** To solve problem with not popular jobs like CyberSecurity and others.We used synthetic data to make dataset more balanced
- **Traceability:** For dividing scrapped data and synthetic data we add columns 'is_custom' and 'source'

## Data Cleaning & Preprocessing
Preprocessing pipeline was made for cleaning and preparing the dataset for models training
- First we remove duplicates and na rows
- Converted skills from javascript to js and others cause of convenience and performance 
- We used regular expressions to extract columns 'experience_level','education_level' and 'skills'
    from unstructured data
- Filled missing data using realistic distribution based on existing data
- Created 'job_id' for each job to show uniqueness of job

## Exploratory Data Analysis (EDA)
Before training, we analyzed the dataset to understand market trends. The visualizations are saved in the `outputs/` folder:
- **Class Distribution:** Confirmed that Data Augmentation successfully balanced our target classes.
- **Average skill per role**: Analyzed the average number of skills per job role.
- **Top 20 Skills:** Identified the most requested technologies across all roles.
- **Experience vs. Education:** Analyzed the correlation between required experience levels and academic degrees.

## Notebook

`src/train_model.ipynb` is kept as a readable demonstration notebook. It shows the same training idea in a notebook format for explanation and presentation.

The reproducible training entry point is:

```bash
python src/train_model.py
```

## Task

The task is supervised multi-class classification. The target column is `label`, which contains the job role class.

The model uses these input features:

- `skills`
- `experience_level`
- `education_level`

The model does not use `job_title`, `label`, `source`, `is_custom`, or `job_id` as input features. Excluding `job_title` is important because it can directly reveal the target class and cause data leakage.

## Feature Engineering

The selected input fields are combined into one text feature called `combined_text`:

```python
combined_text = skills + " " + experience_level + " " + education_level
```

This text is transformed with TF-IDF vectorization. TF-IDF stands for Term Frequency - Inverse Document Frequency. It gives higher weight to terms that are important in one job description but not too common across all job descriptions.

The vectorizer uses unigrams and bigrams with `ngram_range=(1, 2)`. Bigrams help the model learn meaningful phrases such as `machine learning`, `data analysis`, and `ci/cd`.

The TF-IDF vectorizer is fitted only on the training set. The test set is transformed with the already fitted vectorizer, which prevents test-set leakage.

## Models

The training pipeline compares several classical machine learning models:

| Model | Why it is included |
|---|---|
| Dummy Classifier | Baseline model for comparison |
| Multinomial Naive Bayes | Common text classification baseline |
| Complement Naive Bayes | Naive Bayes variant useful for imbalanced text data |
| Logistic Regression | Strong linear classifier for sparse TF-IDF features |
| Linear SVM | Strong classifier for high-dimensional text features |
| Random Forest | Non-linear ensemble model |
| KNN | Nearest-neighbor model connected to course topics |

GridSearchCV is used for the main tuned models:

- Logistic Regression
- Linear SVM
- KNN

The tuning metric is `f1_macro`.

## Evaluation

The dataset is split into train and test sets using an 80/20 stratified split. Stratification keeps a similar class distribution in both parts of the data.

The pipeline reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- 5-fold cross-validation Macro F1

Macro F1 is used as the main model selection metric because this is a multi-class classification problem. It treats all job roles equally and is more reliable than accuracy when class sizes are not exactly the same.

## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | CV Macro F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM Tuned | 0.8893 | 0.8956 | 0.8876 | 0.8890 | 0.8575 |
| Logistic Regression Tuned | 0.8735 | 0.8821 | 0.8719 | 0.8738 | 0.8649 |
| Linear SVM | 0.8617 | 0.8690 | 0.8602 | 0.8612 | 0.8530 |
| Random Forest | 0.8498 | 0.8641 | 0.8465 | 0.8517 | 0.8506 |
| Logistic Regression | 0.8379 | 0.8514 | 0.8405 | 0.8421 | 0.8428 |
| Complement NB | 0.8103 | 0.8547 | 0.8073 | 0.8149 | 0.8035 |
| KNN Tuned | 0.8024 | 0.8158 | 0.8004 | 0.8038 | 0.7796 |
| KNN | 0.7905 | 0.8104 | 0.7925 | 0.7932 | 0.7676 |
| Multinomial NB | 0.7787 | 0.8555 | 0.7699 | 0.7929 | 0.7791 |
| Dummy Most Frequent | 0.1344 | 0.0134 | 0.1000 | 0.0237 | 0.0233 |

## Best Model

The best model is Tuned Linear SVM with `C=5`.

Final test metrics:

- Accuracy: `0.8893`
- Macro Precision: `0.8956`
- Macro Recall: `0.8876`
- Macro F1-score: `0.8890`

Linear SVM performed best because TF-IDF creates sparse high-dimensional text features, and linear SVM is effective for this type of text classification problem.

Although Tuned Logistic Regression had a slightly higher cross-validation Macro F1, Tuned Linear SVM achieved the highest test Macro F1, so it was selected as the final model.

## Saved Artifacts

After running `python src/train_model.py`, the following files are created or updated:

- `models/best_model.pkl`
- `models/vectorizer.pkl`
- `outputs/model_results.csv`
- `outputs/model_comparison.png`
- `outputs/error_analysis.csv`
