# ЗАВИСИМОСТЬ ОТ УЧАСТНИКА 1:
#   Этот файл ожидает готовый data/final_dataset (1).csv от Участника 1.
#   Участник 1 отвечает за: сбор данных, очистку, EDA,
#   нормализацию навыков и сохранение финального датасета.
#   Участник 2 НЕ изменяет датасет, а только использует его
#   для feature engineering и model training.

# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 0 — Импорты и описание роли участника
# ============================================================
# Участник 2 отвечает только за feature engineering и обучение моделей.
# Здесь подключаются библиотеки для работы с таблицами, TF-IDF,
# классическими ML-моделями, метриками, кросс-валидацией и сохранением.

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB


RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "final_dataset (1).csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
RESULTS_PATH = OUTPUTS_DIR / "model_results.csv"
CHART_PATH = OUTPUTS_DIR / "model_comparison.png"
ERROR_ANALYSIS_PATH = OUTPUTS_DIR / "error_analysis.csv"

REQUIRED_COLUMNS = [
    "skills",
    "experience_level",
    "education_level",
    "label",
]

FEATURE_COLUMNS = [
    "skills",
    "experience_level",
    "education_level",
]


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 1 — Загрузка data/final_dataset (1).csv
# ============================================================
# Датасет уже подготовлен Участником 1, поэтому этот код только читает CSV.
# Скрипт не создает заглушки, не выдумывает данные и не изменяет исходный файл.

def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл датасета не найден: {data_path}. "
            "Участник 1 должен подготовить data/final_dataset (1).csv."
        )

    return pd.read_csv(data_path)


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 2 — Проверка обязательных колонок
# ============================================================
# Проверка нужна, чтобы модель обучалась на ожидаемой структуре данных.
# Если колонок не хватает, ошибка явно показывает, что нужно исправить.

def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "В data/final_dataset (1).csv отсутствуют обязательные колонки: "
            + ", ".join(missing_columns)
        )

    missing_labels = df["label"].isna().sum()
    if missing_labels > 0:
        raise ValueError(
            f"В колонке label найдено пустых значений: {missing_labels}. "
            "Для supervised learning каждая строка должна иметь целевой класс."
        )

    class_count = df["label"].nunique()
    if class_count < 2:
        raise ValueError(
            "Для обучения классификатора нужно минимум 2 класса в колонке label."
        )


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 3 — Feature Engineering: combined_text
# ============================================================
# combined_text объединяет только разрешенные признаки:
# skills, experience_level и education_level.
# job_title не используется как feature, чтобы избежать data leakage.
# label, source, is_custom и job_id также не используются как input features.

def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    text_features = df[FEATURE_COLUMNS].fillna("").astype(str)
    df["combined_text"] = (
        text_features.agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    if df["combined_text"].eq("").all():
        raise ValueError(
            "combined_text получился пустым для всех строк. "
            "Проверьте колонки skills, experience_level и education_level."
        )

    return df


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 4 — Train/Test Split
# ============================================================
# Сначала создается combined_text, и только потом выполняется train/test split.
# Это сохраняет test set как unseen data для честной оценки модели.
# Сначала обязательно пробуем stratify=y, чтобы сохранить баланс классов.

def split_dataset(df: pd.DataFrame):
    X_text = df["combined_text"]
    y = df["label"].astype(str)

    try:
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y,
        )
        stratified_split_used = True
    except ValueError as error:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: stratify=y не сработал из-за распределения классов. "
            f"Причина: {error}"
        )
        print("Используется fallback: train_test_split без stratify.")
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        stratified_split_used = False

    return X_train_text, X_test_text, y_train, y_test, stratified_split_used


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 5 — TF-IDF Vectorization после split
# ============================================================
# TF-IDF преобразует текст в числовые признаки для text classification:
# модели sklearn не работают напрямую со строками.
# ngram_range=(1,2) захватывает как отдельные слова ("python"),
# так и пары слов ("machine learning"), что важно для навыков и требований.
# stop_words="english" удаляет частые английские служебные слова из skills.
# В датасете есть много русского текста, поэтому это не универсальная языковая очистка,
# но для текущего проекта такой вариант совпадает с требованием и эмпирически
# дает лучший Macro F1, чем stop_words=None.
# max_features=5000 ограничивает словарь, снижая риск переобучения.
# vectorizer.fit_transform применяется только к train set, потому что
# test set должен оставаться unseen data и не влиять на словарь/IDF веса.

def vectorize_text(X_train_text: pd.Series, X_test_text: pd.Series):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True,
    )

    try:
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
    except ValueError as error:
        raise ValueError(
            "TF-IDF не смог построить признаки. Возможная причина: слишком мало "
            "повторяющихся слов для min_df=2 или пустой текст после обработки."
        ) from error

    return X_train, X_test, vectorizer


# -------------------------------------------------------
# СЛЕДУЮЩИЙ ШАГ → УЧАСТНИК 3 (evaluate_model.py):
#   Загрузи models/best_model.pkl и models/vectorizer.pkl,
#   посчитай confusion matrix, classification report,
#   нарисуй confusion_matrix.png и f1_per_class.png
# -------------------------------------------------------


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 6 — Создание списка моделей
# ============================================================
# Список включает baseline и пять классических ML-моделей.
# DummyClassifier нужен как простой baseline для сравнения качества.

def create_models() -> dict:
    return {
        "Dummy Most Frequent": DummyClassifier(strategy="most_frequent"),
        "Multinomial NB": MultinomialNB(),
        "Complement NB": ComplementNB(), 
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "Linear SVM": LinearSVC(C=1.0, random_state=RANDOM_STATE, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 7 — Safe CV logic
# ============================================================
# 5-fold CV невозможна, если в самом маленьком классе меньше 5 примеров.
# Поэтому cv уменьшается до минимально безопасного значения.
# Если cv < 2, кросс-валидация и GridSearchCV пропускаются с предупреждением.

def build_safe_cv(y_train: pd.Series):
    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min())
    safe_cv = min(5, min_class_count)

    if safe_cv < 2:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: cross-validation пропущена, потому что "
            f"минимальное количество примеров в классе равно {min_class_count}."
        )
        return safe_cv, None

    if safe_cv < 5:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: cv=5 невозможно для текущего train set. "
            f"Используется safe_cv={safe_cv}."
        )

    cv_strategy = StratifiedKFold(
        n_splits=safe_cv,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return safe_cv, cv_strategy


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 8 — Обучение моделей и базовая оценка
# ============================================================
# Каждая модель обучается на X_train и оценивается на X_test.
# Macro Precision, Macro Recall и Macro F1 считаются с zero_division=0,
# чтобы редкие классы не ломали расчет метрик.

def evaluate_predictions(y_test: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro Precision": precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Macro Recall": recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "Macro F1": f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def train_and_evaluate_base_models(models, X_train, X_test, y_train, y_test):
    results = []
    trained_models = {}

    for model_name, model in models.items():
        print(f"Обучение модели: {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred)
        metrics.update(
            {
                "Model": model_name,
                "CV Macro F1": np.nan,
                "Best Params": "",
                "Model Type": "base",
            }
        )

        results.append(metrics)
        trained_models[model_name] = model

    return results, trained_models


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 9 — 5-fold / safe Cross-Validation
# ============================================================
# cross_val_score считает среднее качество на нескольких folds.
# scoring="f1_macro" выбран, потому что проект является multiclass задачей,
# а качество по редким классам важно не меньше общей accuracy.

def add_cross_validation_scores(results, models, X_train, y_train, safe_cv, cv_strategy):
    if safe_cv < 2 or cv_strategy is None:
        for row in results:
            row["CV Macro F1"] = np.nan
        return results

    for row in results:
        model_name = row["Model"]
        print(f"Cross-validation для модели: {model_name}")
        cv_scores = cross_val_score(
            models[model_name],
            X_train,
            y_train,
            cv=cv_strategy,
            scoring="f1_macro",
        )
        row["CV Macro F1"] = cv_scores.mean()

    return results


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 10 — GridSearchCV для Logistic Regression, Linear SVM, KNN
# ============================================================
# GridSearchCV используется только для трех понятных моделей и небольших сеток.
# Это помогает улучшить качество без лишней сложности для защиты проекта.
# Tuning применяется только при safe_cv >= 2.

def tune_selected_models(X_train, X_test, y_train, y_test, safe_cv, cv_strategy):
    tuned_results = []
    tuned_models = {}

    if safe_cv < 2 or cv_strategy is None:
        print(
            "ПРЕДУПРЕЖДЕНИЕ: GridSearchCV пропущен, потому что safe_cv < 2."
        )
        return tuned_results, tuned_models

    tuning_specs = {
        "Logistic Regression Tuned": {
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"),
            "param_grid": {"C": [0.01, 0.1, 1, 5, 10]},
        },
        "Linear SVM Tuned": {
            "estimator": LinearSVC(random_state=RANDOM_STATE,class_weight="balanced"),
            "param_grid": {"C": [0.01, 0.1, 1, 5, 10]},
        },
        "KNN Tuned": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {"n_neighbors": [3, 5, 7, 9]},
        },
    }

    for model_name, spec in tuning_specs.items():
        print(f"GridSearchCV для модели: {model_name}")
        grid_search = GridSearchCV(
            estimator=spec["estimator"],
            param_grid=spec["param_grid"],
            scoring="f1_macro",
            cv=cv_strategy,
        )
        grid_search.fit(X_train, y_train)

        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred)
        metrics.update(
            {
                "Model": model_name,
                "CV Macro F1": grid_search.best_score_,
                "Best Params": grid_search.best_params_,
                "Model Type": "tuned",
            }
        )

        tuned_results.append(metrics)
        tuned_models[model_name] = best_estimator

    return tuned_results, tuned_models


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 11 — Финальная таблица результатов
# ============================================================
# В таблицу попадают обязательные базовые модели, дополнительный Complement NB
# и tuned версии Logistic Regression, Linear SVM и KNN, если GridSearchCV был возможен.

def build_results_table(results: list) -> pd.DataFrame:
    results_df = pd.DataFrame(results)
    results_df = results_df[
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
    return results_df.sort_values(
        by=["Macro F1", "CV Macro F1"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 12 — Выбор best_model по Macro F1
# ============================================================
# Macro F1 используется как главный критерий, потому что он
# оценивает качество по всем классам более равномерно, чем accuracy.
# Если Macro F1 почти одинаковый, дополнительно учитывается CV Macro F1.

def choose_best_model(results_df: pd.DataFrame, trained_models: dict):
    best_macro_f1 = results_df["Macro F1"].max()
    close_candidates = results_df[
        results_df["Macro F1"] >= best_macro_f1 - 0.001
    ].copy()

    close_candidates["CV Rank Score"] = close_candidates["CV Macro F1"].fillna(-1)
    best_row = close_candidates.sort_values(
        by=["CV Rank Score", "Macro F1"],
        ascending=[False, False],
    ).iloc[0]

    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    best_model_macro_f1 = float(best_row["Macro F1"])

    return best_model_name, best_model, best_model_macro_f1


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 13 — Сохранение best_model.pkl, vectorizer.pkl, model_results.csv
# ============================================================
# Перед сохранением создаются папки models/ и outputs/.
# best_model.pkl хранит обученный классификатор, а vectorizer.pkl хранит
# TF-IDF vectorizer, обученный только на train set.

def save_artifacts(best_model, vectorizer, results_df: pd.DataFrame) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    results_df.to_csv(RESULTS_PATH, index=False)


# -------------------------------------------------------
# СЛЕДУЮЩИЙ ШАГ → УЧАСТНИК 3 (evaluate_model.py):
#   Загрузи models/best_model.pkl и models/vectorizer.pkl,
#   посчитай confusion matrix, classification report,
#   нарисуй confusion_matrix.png и f1_per_class.png
# -------------------------------------------------------


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 14 — График outputs/model_comparison.png
# ============================================================
# Bar chart показывает Accuracy и Macro F1 для всех моделей.
# Названия моделей поворачиваются, чтобы подписи не накладывались.

def save_model_comparison_chart(results_df: pd.DataFrame) -> None:
    chart_df = results_df.sort_values("Macro F1", ascending=False).reset_index(drop=True)

    x_positions = np.arange(len(chart_df))
    bar_width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(
        x_positions - bar_width / 2,
        chart_df["Accuracy"],
        width=bar_width,
        label="Accuracy",
    )
    plt.bar(
        x_positions + bar_width / 2,
        chart_df["Macro F1"],
        width=bar_width,
        label="Macro F1",
    )

    plt.xticks(
        x_positions,
        chart_df["Model"],
        rotation=35,
        ha="right",
    )
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Model Comparison: Accuracy vs Macro F1")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 15 — Error Analysis на реальных ошибках
# ============================================================
# Error analysis показывает реальные строки из test set, где best_model ошиблась.
# Это помогает команде понять, какие классы путаются и что улучшать в датасете.

def infer_error_reason(
    true_label: str,
    predicted_label: str,
    combined_text: str,
    class_counts: pd.Series,
) -> str:
    label_pair = {true_label, predicted_label}
    word_count = len(str(combined_text).split())
    rare_classes = set(class_counts[class_counts < 20].index)

    if true_label in rare_classes:
        return "True class has few examples, so the model has limited training signal."
    if label_pair <= {"Backend Developer", "Frontend Developer", "Full Stack Developer"}:
        return "Web development roles share overlapping programming and framework vocabulary."
    if label_pair <= {"Data Analyst", "Data Scientist", "Data Engineer"}:
        return "Data roles share Python, SQL, analytics, and data processing vocabulary."
    if label_pair <= {"Backend Developer", "DevOps Engineer", "Cybersecurity Analyst"}:
        return "Infrastructure, API, Linux, cloud, and security terms can overlap."
    if label_pair <= {"Frontend Developer", "Mobile Developer"}:
        return "Client-side roles can share UI, JavaScript, and application development terms."
    if label_pair <= {"QA Engineer", "Backend Developer", "DevOps Engineer"}:
        return "Testing, API, automation, CI/CD, and backend terms can overlap."
    if word_count < 8:
        return "The text sample is short and may not contain enough distinctive skills."
    return "The sample contains skills that are shared across multiple job roles."


def build_error_analysis(
    df: pd.DataFrame,
    X_test_text: pd.Series,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    base_columns = [
        "job_id",
        "job_title",
        "skills",
        "experience_level",
        "education_level",
    ]
    available_columns = [column for column in base_columns if column in df.columns]
    class_counts = df["label"].astype(str).value_counts()

    error_df = df.loc[X_test_text.index, available_columns].copy()
    error_df["combined_text"] = X_test_text.values
    error_df["true_label"] = y_test.values
    error_df["predicted_label"] = y_pred
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
        error_df = error_df.sort_values(
            by=["true_label", "predicted_label", "job_id"],
            na_position="last",
        ).reset_index(drop=True)
    else:
        error_df["possible_reason"] = pd.Series(dtype=str)

    return error_df


def save_error_analysis(error_analysis_df: pd.DataFrame) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    error_analysis_df.to_csv(ERROR_ANALYSIS_PATH, index=False)


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 16 — Финальный console summary
# ============================================================
# Финальный вывод нужен для защиты проекта: он показывает размер датасета,
# классы, результаты моделей, лучший алгоритм и пути сохраненных файлов.

def print_console_summary(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    best_model_name: str,
    best_model_macro_f1: float,
    error_count: int,
    stratified_split_used: bool,
    safe_cv: int,
) -> None:
    classes = sorted(df["label"].astype(str).unique().tolist())
    printable_results = results_df.copy()
    numeric_columns = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "CV Macro F1",
    ]
    printable_results[numeric_columns] = printable_results[numeric_columns].round(4)

    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЙ ОТЧЕТ УЧАСТНИКА 2 — Feature Engineering + Model Training")
    print("=" * 80)
    print(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Количество классов: {len(classes)}")
    print("Список классов:")
    for class_name in classes:
        print(f"  - {class_name}")
    print(f"Stratified split использован: {stratified_split_used}")
    print(f"Safe CV: {safe_cv if safe_cv >= 2 else 'пропущен'}")
    print("\nТаблица результатов моделей:")
    print(printable_results.to_string(index=False))
    print("\nЛучшая модель:")
    print(f"  Название: {best_model_name}")
    print(f"  Best model Macro F1: {best_model_macro_f1:.4f}")
    print(f"\nКоличество ошибок best model на test set: {error_count}")
    print("\nСохраненные файлы:")
    print(f"  - {BEST_MODEL_PATH}")
    print(f"  - {VECTORIZER_PATH}")
    print(f"  - {RESULTS_PATH}")
    print(f"  - {CHART_PATH}")
    print(f"  - {ERROR_ANALYSIS_PATH}")
    print("=" * 80)


# ============================================================
# [УЧАСТНИК 2] СЕКЦИЯ 17 — Инструкция для Участника 3 в комментарии
# ============================================================
# -------------------------------------------------------
# СЛЕДУЮЩИЙ ШАГ → УЧАСТНИК 3 (evaluate_model.py):
#   Загрузи models/best_model.pkl и models/vectorizer.pkl.
#   Используй тот же порядок обработки нового текста:
#     skills + experience_level + education_level.
#   Преобразуй текст через vectorizer.transform(...).
#   Посчитай confusion matrix, classification report,
#   нарисуй confusion_matrix.png и f1_per_class.png.
#   Не переобучай vectorizer на test/new data.
# -------------------------------------------------------

def main() -> None:
    df = load_dataset(DATA_PATH)
    validate_required_columns(df)
    df = build_combined_text(df)

    X_train_text, X_test_text, y_train, y_test, stratified_split_used = split_dataset(df)
    X_train, X_test, vectorizer = vectorize_text(X_train_text, X_test_text)

    models = create_models()
    safe_cv, cv_strategy = build_safe_cv(y_train)

    base_results, trained_models = train_and_evaluate_base_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    base_results = add_cross_validation_scores(
        base_results,
        models,
        X_train,
        y_train,
        safe_cv,
        cv_strategy,
    )

    tuned_results, tuned_models = tune_selected_models(
        X_train,
        X_test,
        y_train,
        y_test,
        safe_cv,
        cv_strategy,
    )

    all_results = base_results + tuned_results
    trained_models.update(tuned_models)

    results_df = build_results_table(all_results)
    best_model_name, best_model, best_model_macro_f1 = choose_best_model(
        results_df,
        trained_models,
    )
    best_y_pred = best_model.predict(X_test)
    error_analysis_df = build_error_analysis(
        df,
        X_test_text,
        y_test,
        best_y_pred,
    )

    save_artifacts(best_model, vectorizer, results_df)
    save_model_comparison_chart(results_df)
    save_error_analysis(error_analysis_df)

    print_console_summary(
        df,
        results_df,
        best_model_name,
        best_model_macro_f1,
        len(error_analysis_df),
        stratified_split_used,
        safe_cv,
    )


if __name__ == "__main__":
    main()
