import pandas as pd
import random
import uuid

TARGET_COUNTS = {
    "Cybersecurity Analyst": 130,
    "Data Scientist": 80,
    "Data Analyst": 80,
    "Data Engineer": 70,
    "Full Stack Developer": 70,
    "Mobile Developer": 60,
    "Frontend Developer": 60,
}

SKILLS_POOL = {
    "Cybersecurity Analyst": [
        "security",
        "безопасность",
        "пентестер",
        "appsec",
        "soc",
        "siem",
        "owasp",
        "информационная безопасность",
        "уязвимости",
        "kali linux",
        "wireshark",
        "криптография",
        "защита данных",
        "сети",
        "linux",
        "bash",
    ],
    "Data Scientist": [
        "machine learning",
        "ml",
        "data science",
        "python",
        "pandas",
        "computer vision",
        "nlp",
        "нейросети",
        "deep learning",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "математика",
        "статистика",
    ],
    "Data Analyst": [
        "data analyst",
        "аналитик данных",
        "bi аналитик",
        "sql",
        "excel",
        "tableau",
        "power bi",
        "python",
        "pandas",
        "a/b тестирование",
        "анализ данных",
        "базы данных",
        "google analytics",
    ],
    "Data Engineer": [
        "data engineer",
        "инженер данных",
        "dwh",
        "etl",
        "python",
        "sql",
        "spark",
        "hadoop",
        "airflow",
        "kafka",
        "postgresql",
        "базы данных",
        "docker",
        "kubernetes",
    ],
    "Full Stack Developer": [
        "full stack",
        "фуллстек",
        "javascript",
        "typescript",
        "react",
        "node.js",
        "python",
        "django",
        "php",
        "postgresql",
        "docker",
        "html",
        "css",
        "git",
    ],
    "Mobile Developer": [
        "mobile",
        "ios",
        "android",
        "flutter",
        "swift",
        "kotlin",
        "мобильный разработчик",
        "react native",
        "rest api",
        "мобильные приложения",
        "ui/ux",
    ],
    "Frontend Developer": [
        "frontend",
        "фронтенд",
        "javascript",
        "typescript",
        "react",
        "vue",
        "angular",
        "html",
        "css",
        "sass",
        "figma",
        "ui/ux",
        "webpack",
        "spa",
    ],
}

EXP_LEVELS = [
    "Без опыта",
    "Опыт работы от 1 до 3 лет",
    "Опыт работы от 3 до 6 лет",
    "Опыт более 6 лет",
]
EDU_LEVELS = [
    "Высшее техническое образование",
    "Высшее образование",
    "Среднее специальное образование",
    "Студент",
    "Не имеет значения",
]

TEMPLATES = [
    "В международную компанию требуется {title}. Обязанности включают работу с технологиями: {skills}. Мы ожидаем, что ваш опыт: {exp}. Требуемое образование: {edu}. Предлагаем ДМС, гибкий график и отличный коллектив.",
    "Ищем сильного {title} для участия в высоконагруженных проектах. Наш стек: {skills}. Пожелания к кандидату: {edu}, стаж работы: {exp}.",
    "Стартап расширяет команду. Открыта вакансия {title}. Что нужно знать: {skills}. Условия: {exp}, {edu}. Полная удаленка, печеньки в офисе.",
    "Крупный банк ищет специалиста на позицию {title}. Задачи: поддержка и развитие продукта. Требования: {skills}. Ожидаемый опыт: {exp}, {edu}. Официальное трудоустройство.",
    "{title} в команду продуктовой разработки. Ждем от вас уверенных знаний: {skills}. Требования к кандидату: {edu}, {exp}. Конкурентная зарплата и премии.",
]


def generate_synthetic_data():
    data = []

    for role, count in TARGET_COUNTS.items():
        pool = SKILLS_POOL[role]
        for _ in range(count):
            num_skills = random.randint(4, min(8, len(pool)))
            selected_skills = ", ".join(random.sample(pool, num_skills))

            exp = random.choice(EXP_LEVELS)
            edu = random.choice(EDU_LEVELS)

            data.append(
                {
                    "job_id": str(uuid.uuid4())[:8],
                    "job_title": role,
                    "skills": selected_skills,
                    "experience_level": exp,
                    "education_level": edu,
                    "label": role,
                    "source": "synthetic_generator",
                    "is_custom": "yes",
                }
            )

    df = pd.DataFrame(data)

    output_path = "../../data/custom_jobs.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Generated {len(df)} synthetic data")
    print(f"File saved to: {output_path}")


if __name__ == "__main__":
    generate_synthetic_data()
