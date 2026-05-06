import pandas as pd
import re
import uuid
import random

KNOWN_SKILLS = [
    'python', 'java', 'javascript', 'typescript', 'c#', '.net', 'c++', 'go', 'golang', 'php', 'ruby',
    'react', 'angular', 'vue', 'next.js', 'nuxt', 'node.js', 'express', 'fastapi', 'django', 'spring',
    'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'kafka', 'rabbitmq',
    'docker', 'kubernetes', 'k8s', 'ci/cd', 'git', 'linux', 'bash', 'ansible', 'terraform',
    'machine learning', 'ml', 'data science', 'pandas', 'numpy', 'scikit-learn', 'pytorch', 'tensorflow',
    'html', 'css', 'sass', 'tailwind', 'figma', 'ui/ux', 'cypress', 'jest'
]


def extract_skills(text):
    if pd.isna(text): return ""
    text_lower = str(text).lower()
    
    text_lower = text_lower.replace('js', 'javascript').replace('ds', 'data science')
    text_lower = text_lower.replace('машинное обучение', 'machine learning')
    
    found = [skill for skill in KNOWN_SKILLS if re.search(fr'\b{re.escape(skill)}\b', text_lower)]
    return ", ".join(found) 

def extract_experience(text):
    if pd.isna(text): return "Not Specified"
    text_lower = str(text).lower()
    if re.search(r'без опыта', text_lower): return 'Без опыта'
    if re.search(r'от 1( года)? до 3', text_lower): return 'От 1 года до 3 лет'
    if re.search(r'от 3 до 6', text_lower): return 'От 3 до 6 лет'
    if re.search(r'более 6', text_lower) or re.search(r'\b6\+ лет', text_lower): return 'Более 6 лет'
    return "Not Specified"

def extract_education(text):
    if pd.isna(text): return "Not Specified"
    text_lower = str(text).lower()
    if re.search(r'высшее', text_lower): return 'Высшее образование'
    if re.search(r'среднее', text_lower): return 'Среднее образование'
    if re.search(r'студент', text_lower): return 'Студент'
    return "Not Specified"

def fill_random_if_missing(row):
    exp_choices = ['Без опыта', 'От 1 года до 3 лет', 'От 1 года до 3 лет', 'От 3 до 6 лет', 'Более 6 лет']
    edu_choices = ['Высшее образование', 'Высшее образование', 'Среднее образование', 'Студент']
    
    if row['experience_level'] == "Not Specified":
        row['experience_level'] = random.choice(exp_choices)
        
    if row['education_level'] == "NotSpecified":
        row['education_level'] = random.choice(edu_choices)
        
    return row

def classify_source(source_val):
    if pd.isna(source_val):
        return 'other'
    
    s = str(source_val).lower()
    
    if 'hh.kz' in s:
        return 'hh'
    elif 'qyzmet.kz' in s:
        return 'qyzmet'
    elif 'synthetic' in s:
        return 'synthetic'
    
    return 'other'


def assign_label(title):
    title = str(title).lower()
    if any(word in title for word in ['full stack', 'full-stack', 'фуллстек', 'фулстек']): return 'Full Stack Developer'
    if any(word in title for word in ['qa', 'tester', 'тестировщик', 'quality assurance', 'автотестировщик']): return 'QA Engineer'
    if any(word in title for word in ['data scientist', 'machine learning', 'ml', 'data science', 'computer vision', 'nlp']): return 'Data Scientist'
    if any(word in title for word in ['data analyst', 'аналитик данных', 'bi аналитик', 'bi-аналитик']): return 'Data Analyst'
    if any(word in title for word in ['data engineer', 'инженер данных', 'dwh', 'etl']): return 'Data Engineer'
    if any(word in title for word in ['cybersecurity', 'security', 'безопасност', 'пентестер', 'appsec', 'soc']): return 'Cybersecurity Analyst'
    if any(word in title for word in ['devops', 'sre', 'инфраструктур', 'kubernetes', 'системный администратор']): return 'DevOps Engineer'
    if any(word in title for word in ['mobile', 'ios', 'android', 'flutter', 'swift', 'kotlin', 'мобильный']): return 'Mobile Developer'
    if any(word in title for word in ['frontend', 'front-end', 'фронтенд', 'react', 'vue', 'angular']): return 'Frontend Developer'
    if any(word in title for word in ['backend', 'back-end', 'бэкенд', 'python', 'java', 'php', 'c#', '.net', 'golang', 'c++', 'ruby', 'node', 'сервер']): return 'Backend Developer'
    return 'Other'


def main():
    
    df = pd.read_csv('../data/raw/raw_jobs.csv', sep=',', low_memory=False) 
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'title' in df.columns and 'job_title' in df.columns:
        df['job_title'] = df['job_title'].fillna(df['title'])
    elif 'title' in df.columns:
        df = df.rename(columns={'title': 'job_title'})

    req_col = 'requirements' if 'requirements' in df.columns else 'skills'
    resp_col = 'responsibilities' if 'responsibilities' in df.columns else 'qualifications'
    df['raw_description'] = df[req_col].fillna('') + " " + df[resp_col].fillna('')
    
    df['label'] = df['job_title'].astype(str).apply(assign_label)
    df = df[df['label'] != 'Other']
    
    initial_len = len(df)
    df = df.drop_duplicates(subset=['raw_description'])
    print(f"Deleting Duplictes {initial_len - len(df)}")

    print("Extracting skills, experience, education")
    df['skills'] = df['raw_description'].apply(extract_skills)
    df['experience_level'] = df['raw_description'].apply(extract_experience)
    df['education_level'] = df['raw_description'].apply(extract_education)
    

    df = df[df['skills'].str.len() > 0]
    

    df = df.apply(fill_random_if_missing, axis=1)
    
    df['job_id'] = [str(uuid.uuid4())[:8] for _ in range(len(df))]

    df['source'] = df['url'].apply(classify_source)
    df['is_custom'] = df['source'].apply(lambda x: 'yes' if x == 'synthetic' else 'no')

    final_columns = ['job_id', 'job_title', 'skills', 'experience_level', 'education_level', 'label', 'source', 'is_custom']
    df_final = df[final_columns]
    
    print("\n Balance of classes:")
    class_counts = df_final['label'].value_counts()
    print(class_counts)
    
    df_final.to_csv('../data/final_dataset.csv', index=False, encoding='utf-8-sig')
    print("\n Final dataset saved to 'data/final_dataset.csv'!")

if __name__ == "__main__":
    main()