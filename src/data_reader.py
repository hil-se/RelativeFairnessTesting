import pandas as pd
import numpy as np

def load_adult():
    data = pd.read_csv('../data/adult.csv')
    # Drop columns with missing values
    # missing = [key for key in data.keys() if data[key].dtype == 'O']
    # data = data.drop(missing, axis=1)
    data['sex'] = data['sex'].apply(lambda x: 1 if x == "Male" else 0)
    # discretize race: white vs. non-white
    data['race'] = data['race'].apply(lambda x: 1 if x == "White" else 0)
    # prefer >50K as label 1
    data['income'] = data['income'].apply(lambda x: 1 if x == ">50K" else 0)
    # Separate independent variables and dependent variables
    dependent = 'income'
    X = data.drop(dependent, axis=1)
    y = np.array(data[dependent])
    protected = ['sex', 'race']
    return X, y, protected

def load_heart():
    df = pd.read_csv("../data/heart.csv")
    # sensitive attribute names
    A = ["age"]
    # discretize age: x>60
    df['age'] = df['age'].apply(lambda x: 1 if x > 60 else 0)
    # prefer 0 (< 50% diameter narrowing) as label 1
    df['y'] = df['y'].apply(lambda x: 1 if x==0 else 0)
    dependent = 'y'
    X = df.drop(dependent, axis=1)
    y = np.array(df[dependent])
    return X, y, A

def load_default():
    df = pd.read_csv("../data/default.csv")
    # sensitive attribute names
    A = ["SEX"]
    df['SEX'] = df['SEX'].apply(lambda x: 0 if x == 2 else 1)
    # prefer 0 (Default Payment = No) as label 1
    df['default payment next month'] = df['default payment next month'].apply(lambda x: 1 if x == 0 else 0)
    dependent = 'default payment next month'
    X = df.drop(dependent, axis=1)
    y = np.array(df[dependent])
    return X, y, A

def load_compas():
    df = pd.read_csv("../data/compas-scores-two-years.csv")
    features_to_keep = ['sex', 'age', 'age_cat', 'race',
                        'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                        'priors_count', 'c_charge_degree', 'c_charge_desc',
                        'two_year_recid']
    df = df[features_to_keep]
    # sensitive attribute names
    A = ["sex", "race"]
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "Male" else 0)
    # discretize race: Caucasian vs. non-Caucasian
    df['race'] = df['race'].apply(lambda x: 1 if x == "Caucasian" else 0)
    # prefer 0 (no recid) as label 1
    df['two_year_recid'] = df['two_year_recid'].apply(lambda x: 1 if x==0 else 0)
    dependent = 'two_year_recid'
    X = df.drop(dependent, axis=1)
    y = np.array(df[dependent])
    return X, y, A

def load_bank():
    df = pd.read_csv("../data/bank.csv", sep =";")
    # sensitive attribute names
    A = ["age"]
    # discretize age: x>25
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    # prefer yes as label 1
    df['y'] = df['y'].apply(lambda x: 1 if x=="yes" else 0)
    dependent = 'y'
    X = df.drop(dependent, axis=1)
    y = np.array(df[dependent])
    return X, y, A

def load_german():
    column_names = ['status', 'month', 'credit_history',
                    'purpose', 'credit_amount', 'savings', 'employment',
                    'investment_as_income_percentage', 'sex',
                    'other_debtors', 'residence_since', 'property', 'age',
                    'installment_plans', 'housing', 'number_of_credits',
                    'skill_level', 'people_liable_for', 'telephone',
                    'foreign_worker', 'credit']
    df = pd.read_csv("../data/german.data", sep=' ', header=None, names=column_names)
    # sensitive attribute names
    A = ["age", "sex"]
    # discretize age: x>25
    df["age"] = df["age"].apply(lambda x: 1 if x > 25 else 0)
    # transform personal_status into sex
    df["sex"] = df["sex"].apply(lambda x: 1 if x in {"A91", "A93", "A94"} else 0)
    # prefer 1 (good credit) as label 1
    df['credit'] = df['credit'].apply(lambda x: 1 if x==1 else 0)
    dependent = 'credit'
    X = df.drop(dependent, axis=1)
    y = np.array(df[dependent])
    return X, y, A

def load_student_mat():
    df = pd.read_csv("../data/student-mat.csv", sep=";")
    # sensitive attribute names
    A = ["sex"]
    # discretize sex
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "M" else 0)
    # prefer yes as label 1
    df['y'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G3'])
    dependent = 'y'
    X = df.drop(dependent, axis=1)
    y = np.array(df[dependent])
    return X, y, A

def load_student_por():
    df = pd.read_csv("../data/student-por.csv", sep=";")
    # sensitive attribute names
    A = ["sex"]
    # discretize sex
    df['sex'] = df['sex'].apply(lambda x: 1 if x == "M" else 0)
    # prefer yes as label 1
    df['y'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df = df.drop(columns=['G3'])
    dependent = 'y'
    X = df.drop(dependent, axis=1)
    y = np.array(df[dependent])
    return X, y, A


