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

## How to Run Website

The website app is not included in this `ML` folder yet.

When `website/app.py` is added, the expected command will be:

```bash
streamlit run website/app.py
```

The website should load:

- `models/best_model.pkl`
- `models/vectorizer.pkl`

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

## Public Links

- GitHub repository: TODO
- Deployed website: TODO
- Public video demo: TODO
- Final report: TODO
- Poster: TODO
