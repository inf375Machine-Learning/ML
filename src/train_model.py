"""Train the job role classifier and to save reproducible artifacts."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "final_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
RESULTS_PATH = OUTPUTS_DIR / "model_results.csv"
CHART_PATH = OUTPUTS_DIR / "model_comparison.png"
ERROR_ANALYSIS_PATH = OUTPUTS_DIR / "error_analysis.csv"

FEATURE_COLUMNS = ["skills", "experience_level", "education_level"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["label"]


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if df["label"].isna().any():
        raise ValueError(f"Found {df['label'].isna().sum()} empty labels")

    print(f"Dataset: {df.shape[0]} rows, {df['label'].nunique()} classes")
    print(df["label"].value_counts())
    return df


def build_text_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    text_features = df[FEATURE_COLUMNS].fillna("").astype(str)
    combined_text = (
        text_features.agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["combined_text"] = combined_text
    labels = df["label"].astype(str)
    return combined_text, labels


def make_models() -> dict[str, object]:
    return {
        "Dummy Most Frequent": DummyClassifier(strategy="most_frequent"),
        "Multinomial NB": MultinomialNB(),
        "Complement NB": ComplementNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Linear SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


def eval_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_base_models(
    models: dict[str, object],
    X_train,
    X_test,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_strategy,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = eval_metrics(y_test, y_pred)
        cv_f1 = (
            cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="f1_macro").mean()
            if cv_strategy
            else np.nan
        )
        metrics.update(
            {
                "Model": name,
                "CV Macro F1": cv_f1,
                "Best Params": "",
                "Model Type": "base",
            }
        )
        results.append(metrics)
        trained_models[name] = model
        print(
            f"{name:25s} | Acc={metrics['Accuracy']:.4f} | "
            f"F1={metrics['Macro F1']:.4f} | CV={cv_f1:.4f}"
        )

    return results, trained_models


def tune_models(
    X_train,
    X_test,
    y_train: pd.Series,
    y_test: pd.Series,
    cv_strategy,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not cv_strategy:
        return [], {}

    tuning_specs = {
        "Logistic Regression Tuned": {
            "estimator": LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            "param_grid": {"C": [0.01, 0.1, 1, 5, 10]},
        },
        "Linear SVM Tuned": {
            "estimator": LinearSVC(random_state=RANDOM_STATE, class_weight="balanced"),
            "param_grid": {"C": [0.01, 0.1, 1, 5, 10]},
        },
        "KNN Tuned": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {"n_neighbors": [3, 5, 7, 9]},
        },
    }

    results = []
    trained_models = {}

    for name, spec in tuning_specs.items():
        search = GridSearchCV(spec["estimator"], spec["param_grid"], scoring="f1_macro", cv=cv_strategy)
        search.fit(X_train, y_train)
        y_pred = search.best_estimator_.predict(X_test)
        metrics = eval_metrics(y_test, y_pred)
        metrics.update(
            {
                "Model": name,
                "CV Macro F1": search.best_score_,
                "Best Params": search.best_params_,
                "Model Type": "tuned",
            }
        )
        results.append(metrics)
        trained_models[name] = search.best_estimator_
        print(
            f"{name:25s} | Acc={metrics['Accuracy']:.4f} | "
            f"F1={metrics['Macro F1']:.4f} | CV={search.best_score_:.4f} | "
            f"{search.best_params_}"
        )

    return results, trained_models


def build_results_table(results: list[dict[str, object]]) -> pd.DataFrame:
    return (
        pd.DataFrame(results)[
            [
                "Model",
                "Model Type",
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1",
                "CV Macro F1",
                "Best Params",
            ]
        ]
        .sort_values(["Macro F1", "CV Macro F1"], ascending=False)
        .reset_index(drop=True)
    )


def select_best_model(results_df: pd.DataFrame, trained_models: dict[str, object]) -> tuple[str, object]:
    top_f1 = results_df["Macro F1"].max()
    candidates = results_df[results_df["Macro F1"] >= top_f1 - 0.001].copy()
    candidates["_cv"] = candidates["CV Macro F1"].fillna(-1)
    best_row = candidates.sort_values(["_cv", "Macro F1"], ascending=False).iloc[0]
    best_name = best_row["Model"]
    print(f"\nBest model: {best_name} (Macro F1: {best_row['Macro F1']:.4f})")
    return best_name, trained_models[best_name]


def save_model_comparison_chart(results_df: pd.DataFrame) -> None:
    chart_df = results_df.sort_values("Macro F1", ascending=False).reset_index(drop=True)
    x_positions = np.arange(len(chart_df))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x_positions - width / 2, chart_df["Accuracy"], width=width, label="Accuracy")
    plt.bar(x_positions + width / 2, chart_df["Macro F1"], width=width, label="Macro F1")
    plt.xticks(x_positions, chart_df["Model"], rotation=35, ha="right")
    plt.ylabel("Score")
    plt.title("Model Comparison: Accuracy vs Macro F1")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()


def infer_error_reason(true_label: str, predicted_label: str, text: str, class_counts: pd.Series) -> str:
    label_pair = {true_label, predicted_label}
    rare_classes = set(class_counts[class_counts < 20].index)
    if true_label in rare_classes:
        return "True class has few examples"
    if label_pair <= {"Backend Developer", "Frontend Developer", "Full Stack Developer"}:
        return "Overlapping web dev vocabulary"
    if label_pair <= {"Data Analyst", "Data Scientist", "Data Engineer"}:
        return "Overlapping data role vocabulary"
    if label_pair <= {"Backend Developer", "DevOps Engineer", "Cybersecurity Analyst"}:
        return "Overlapping infra/security terms"
    if label_pair <= {"Frontend Developer", "Mobile Developer"}:
        return "Overlapping client-side terms"
    if label_pair <= {"QA Engineer", "Backend Developer", "DevOps Engineer"}:
        return "Overlapping testing/backend terms"
    if len(str(text).split()) < 8:
        return "Short text sample"
    return "Skills shared across multiple roles"


def build_error_analysis(
    df: pd.DataFrame,
    X_test_text: pd.Series,
    y_test: pd.Series,
    best_y_pred: np.ndarray,
) -> pd.DataFrame:
    base_cols = [
        column
        for column in ["job_id", "job_title", "skills", "experience_level", "education_level"]
        if column in df.columns
    ]
    class_counts = df["label"].astype(str).value_counts()

    error_df = df.loc[X_test_text.index, base_cols].copy()
    error_df["combined_text"] = X_test_text.values
    error_df["true_label"] = y_test.values
    error_df["predicted_label"] = best_y_pred
    error_df = error_df[error_df["true_label"] != error_df["predicted_label"]].copy()
    if not error_df.empty:
        error_df["possible_reason"] = error_df.apply(
            lambda row: infer_error_reason(
                row["true_label"],
                row["predicted_label"],
                row["combined_text"],
                class_counts,
            ),
            axis=1,
        )

    print(f"Errors: {len(error_df)} / {len(y_test)} test samples")
    return error_df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X_text, y = build_text_features(df)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    min_class_count = int(y_train.value_counts().min())
    safe_cv = min(5, min_class_count)
    cv_strategy = (
        StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=RANDOM_STATE)
        if safe_cv >= 2
        else None
    )
    print(f"CV folds: {safe_cv}")

    base_results, trained_models = train_base_models(
        make_models(),
        X_train,
        X_test,
        y_train,
        y_test,
        cv_strategy,
    )
    tuned_results, tuned_models = tune_models(X_train, X_test, y_train, y_test, cv_strategy)
    all_results = base_results + tuned_results
    trained_models.update(tuned_models)

    results_df = build_results_table(all_results)
    print(results_df)
    _, best_model = select_best_model(results_df, trained_models)

    save_model_comparison_chart(results_df)
    best_y_pred = best_model.predict(X_test)
    error_df = build_error_analysis(df, X_test_text, y_test, best_y_pred)

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    results_df.to_csv(RESULTS_PATH, index=False)
    error_df.to_csv(ERROR_ANALYSIS_PATH, index=False)

    print(f"Saved: {BEST_MODEL_PATH}")
    print(f"Saved: {VECTORIZER_PATH}")
    print(f"Saved: {RESULTS_PATH}")
    print(f"Saved: {CHART_PATH}")
    print(f"Saved: {ERROR_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
