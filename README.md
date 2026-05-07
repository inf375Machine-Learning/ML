# Skill Job Role Predictor

## Short Project Description

This project predicts a suitable IT job role based on a user's skills, education level, and experience level.

The machine learning pipeline uses TF-IDF feature extraction and supervised multi-class classification. The final model predicts one of several job roles, such as Backend Developer, Frontend Developer, Data Analyst, Data Scientist, DevOps Engineer, QA Engineer, Mobile Developer, Full Stack Developer, Data Engineer, or Cybersecurity Analyst.

`src/train_model.py` is the reproducible training entry point. `src/train_model.ipynb` is kept as a readable notebook for demonstration and explanation.

## How to Install Dependencies

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

If you need to run the parser scripts, install Playwright browsers:

```bash
playwright install
```

## How to Run Training

Run the reproducible training pipeline:

```bash
python src/train_model.py
```

The script can be launched from any working directory because paths are resolved relative to the script location.

The training pipeline:

- loads `data/final_dataset.csv`
- combines `skills`, `experience_level`, and `education_level`
- converts text into TF-IDF features
- trains and compares multiple models
- uses Macro F1-score for model selection
- saves the best model and output files

Best model: **Tuned Linear SVM**

Main result:

- Accuracy: `0.8893`
- Macro Precision: `0.8956`
- Macro Recall: `0.8876`
- Macro F1-score: `0.8890`

## How to Run the Website

The Streamlit web demo is available as a public deployment and can also be run locally.

Public URL: https://role-finder.streamlit.app

To run the website locally from the project root:

```bash
streamlit run website/app.py
```

The website loads `models/best_model.pkl` and `models/vectorizer.pkl`, accepts free-text skills together with experience and education levels through a clean form, and returns the predicted job role in a result card.

## How to Run Evaluation

The evaluation notebook reproduces the saved test split, recomputes overall and per-class metrics, and writes the confusion matrix, the per-class precision/recall/F1 chart, the classification report, and the overall metrics table to `outputs/`. Open it with:

```bash
jupyter notebook src/evaluation/evaluation.ipynb
```

The notebook reuses the saved `best_model.pkl` and `vectorizer.pkl`, recreates the same stratified split with `random_state=42`, and reports Accuracy, Macro Precision, Macro Recall, Macro F1, a 5-fold cross-validation summary, a data leakage prevention note, and a qualitative interpretation of the saved error analysis.

## How to Use the Prediction Function

The prediction wrapper exposes a single function that loads the cached model and vectorizer once and returns the predicted job role for any skill text:

```python
from src.predict import predict_job_role

role = predict_job_role(skills="Python, SQL, pandas, statistics, Power BI")
print(role)
```

The same function is used internally by the Streamlit web app to keep the inference logic decoupled from the user interface.

## Where Outputs Are Saved

Model artifacts are saved in:

```text
models/
```

Saved model files:

- `models/best_model.pkl`
- `models/vectorizer.pkl`

Training results and visualizations are saved in:

```text
outputs/
```

Main output files:

- `outputs/model_results.csv`
- `outputs/model_comparison.png`
- `outputs/error_analysis.csv`
- `outputs/class_distribution.png`
- `outputs/top_20_skills.png`
- `outputs/average_skills_per_role.png`
- `outputs/custom_vs_public.png`
- `outputs/confusion_matrix.png`
- `outputs/f1_per_class.png`
- `outputs/classification_report.txt`
- `outputs/overall_metrics.csv`

## Public Links

- GitHub repository: https://github.com/inf375Machine-Learning/ML
- Deployed website: https://role-finder.streamlit.app
- Public video demo: TODO
- Final report: https://docs.google.com/document/d/1bB0f_33X26WGMqgmdkxcm-CfOfke3tberf-D3PR2vFU/edit?tab=t.0
- Poster: https://canva.link/ow21ywt65a0t0uo

## Connection to INF375 Course Topics

This project applies several concepts covered during the semester:

- **Supervised learning and classification.** Each row in the dataset has input features (skills, experience level, education level) and a known categorical output (job role), and the model learns a mapping from inputs to labels.
- **Text categorization with bag-of-words and n-grams.** Skills and qualifications are text data, vectorized with TF-IDF using unigrams and bigrams (`ngram_range=(1, 2)`).
- **Multiple classical algorithms compared.** The pipeline trains and evaluates Naive Bayes (Multinomial and Complement), K-Nearest Neighbors, Logistic Regression, Linear Support Vector Machine, and Random Forest, alongside a Dummy baseline.
- **Hyperparameter tuning.** GridSearchCV is used for Logistic Regression, Linear SVM, and KNN with `scoring="f1_macro"`.
- **Evaluation beyond accuracy.** Macro Precision, Macro Recall, Macro F1, classification report, confusion matrix, and 5-fold cross-validation are reported.
- **Overfitting prevention.** Stratified train/test split, stratified cross-validation, regularization in linear models, and limited TF-IDF vocabulary (`max_features=5000`, `min_df=2`) keep the model from memorizing the training set.

## Team Contribution

The team is composed of three members. Each member owned one module end-to-end, and all members participated in dataset review, model discussion, poster preparation, video recording, and website testing.

- **Team Member 1.** Dataset collection, cleaning, exploratory data analysis, and dataset documentation. Wrote the parsers in `src/parser/` and the preprocessing scripts in `src/preproc/`. Produced the EDA notebook in `src/graphics/EDA.ipynb` and the dataset section of this README.
- **Team Member 2.** Feature engineering and model training. Wrote `src/train_model.py` and `src/train_model.ipynb`, performed the GridSearchCV tuning, selected the best model by Macro F1, and saved `models/best_model.pkl` and `models/vectorizer.pkl`. Authored the algorithm and training sections of this README.
- **Team Member 3.** Model evaluation, prediction wrapper, web demo, and project deliverables. Wrote `src/evaluation/evaluation.ipynb` (confusion matrix, per-class metrics, cross-validation summary, error analysis interpretation), `src/predict.py`, and `website/app.py`. Deployed the public Streamlit web app, assembled the poster and the final report, and authored the implementation, links, limitations, and team contribution sections of this README.

## Limitations and Ethical Considerations

- The model recommends a job role based only on the provided skills, experience level, and education level, and should be used as a career guidance tool rather than as a hiring or admission decision system.
- Predictions reflect the dataset and may carry the biases present in the public job descriptions and the manually labeled samples that were used during training.
- Several job roles share substantial vocabulary (for example Backend Developer with Full Stack Developer, or Data Analyst with Data Scientist and Data Engineer), so the model can confuse these pairs when the skill list is short or contains terms common to both.
- Typing errors are not corrected automatically, because TF-IDF treats tokens as exact matches. A few misspellings have little effect when the rest of the input is intact, but heavily misspelled inputs reduce prediction quality.
- The deployed website is a class demonstration of the trained model and is not a professional career assessment platform.
