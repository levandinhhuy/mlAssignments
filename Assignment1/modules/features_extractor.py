import numpy as np
import pandas as pd

# EXPERIENCE FEATURES
def fe_experience_years(df):
    def map_func(x):
        if x == 0:   # >20
            return 21
        elif x == 1: # 5
            return 5
        elif x == 2: # 4
            return 4
        else:
            return 2
    df['experience_years'] = df['experience'].apply(map_func)
    return df

# LAST NEW JOB FEATURES
def fe_last_new_job_gap(df):
    def map_func(x):
        if x == 0:   # 1
            return 1
        elif x == 1: # >4
            return 5
        elif x == 2: # 2
            return 2
        else:
            return 3
    df['last_new_job_gap'] = df['last_new_job'].apply(map_func)
    return df

# TRAINING FEATURES
def fe_training_features(df):
    df['training_hours_high'] = (df['training_hours'] > 80).astype(int)
    df['training_hours_per_experience'] = df['training_hours'] / (df['experience_years'] + 1)
    return df

# EDUCATION FEATURES
def fe_education_features(df):
    df['education_level_high'] = df['education_level'].isin([1, 4]).astype(int)
    df['education_level_low'] = (df['education_level'] == 2).astype(int)
    return df

# MAJOR FEATURES
def fe_major_features(df):
    df['major_discipline_stem'] = (df['major_discipline'] == 0).astype(int)
    df['major_discipline_business'] = (df['major_discipline'] == 3).astype(int)
    return df

# UNIVERSITY FEATURES
def fe_university_features(df):
    df['enrolled_university_active'] = df['enrolled_university'].isin([1, 2]).astype(int)
    return df

# CITY FEATURES
def fe_city_features(df):
    df['city_development_high'] = (df['city_development_index'] > 0.8).astype(int)
    return df

# INTERACTION FEATURES
def fe_interaction_features(df):
    df['education_experience_mismatch'] = (
        (df['education_level_high'] == 1) & (df['experience_years'] < 3)
    ).astype(int)

    df['career_change_score'] = (
        (df['last_new_job_gap'] <= 1).astype(int) +
        df['training_hours_high'] +
        (df['relevent_experience'] == 0).astype(int)
    )

    return df

# MAIN PIPELINE FUNCTION
def apply_feature_extraction(df):
    df = df.copy()

    df = fe_experience_years(df)
    df = fe_last_new_job_gap(df)

    df = fe_training_features(df)
    df = fe_education_features(df)
    df = fe_major_features(df)
    df = fe_university_features(df)
    df = fe_city_features(df)

    df = fe_interaction_features(df)

    return df