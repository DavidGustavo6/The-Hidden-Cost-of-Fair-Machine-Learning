"""
Multi-dataset loader for fairness experiments.
This module provides standardized loaders for common fairness benchmark datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Optional, Tuple
import warnings
import os
warnings.filterwarnings('ignore')


def load_german_credit(
    data_path: Optional[str] = None,
    sensitive_attr: str = 'sex',
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict:
    """
    Load the German Credit Dataset.
    
    The German Credit dataset classifies people as good or bad credit risks.
    
    Parameters
    ----------
    data_path : str, optional
        Path to local file. If None, downloads from UCI.
    sensitive_attr : str
        Sensitive attribute: 'sex' or 'age'
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
        
    Returns
    -------
    Dict with train/test data and metadata
    """
    if data_path is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        columns = [
            'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
            'savings', 'employment', 'installment_rate', 'personal_status_sex',
            'other_debtors', 'residence', 'property', 'age', 'other_plans',
            'housing', 'existing_credits', 'job', 'num_dependents', 'telephone', 'foreign_worker', 'target'
        ]
        df = pd.read_csv(url, sep=' ', header=None, names=columns)
    else:
        df = pd.read_csv(data_path)
    
    # Extract sex from personal_status_sex (A91=male divorced, A92=female divorced/married, 
    # A93=male single, A94=male married, A95=female single)
    df['sex'] = df['personal_status_sex'].apply(
        lambda x: 1 if x in ['A91', 'A93', 'A94'] else 0  # 1=male, 0=female
    )
    
    # Create age binary (young < 25, old >= 25)
    df['age_binary'] = (df['age'] >= 25).astype(int)
    
    # Target: 1=Good, 2=Bad -> Convert to 1=Good, 0=Bad
    df['target'] = (df['target'] == 1).astype(int)
    
    # Encode categorical variables
    categorical_cols = ['status', 'credit_history', 'purpose', 'savings', 'employment',
                       'personal_status_sex', 'other_debtors', 'property', 'other_plans',
                       'housing', 'job', 'telephone', 'foreign_worker']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Select sensitive attribute
    if sensitive_attr == 'sex':
        sensitive_col = 'sex'
    elif sensitive_attr == 'age':
        sensitive_col = 'age_binary'
    else:
        raise ValueError(f"Unknown sensitive attribute: {sensitive_attr}")
    
    # Prepare features and target
    feature_cols = [c for c in df.columns if c not in ['target', 'sex', 'age_binary']]
    X = df[feature_cols].values
    y = df['target'].values
    sensitive = df[sensitive_col].values
    
    # Split data
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'sensitive_train': sens_train,
        'sensitive_test': sens_test,
        'sensitive_attr': sensitive_attr,
        'feature_names': feature_cols,
        'scaler': scaler,
        'encoders': encoders,
        'metadata': {
            'name': 'German Credit',
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'task': 'Credit risk classification',
            'positive_label': 'Good credit',
            'sensitive_groups': {0: 'Female' if sensitive_attr == 'sex' else 'Young', 
                               1: 'Male' if sensitive_attr == 'sex' else 'Old'}
        }
    }


def load_compas(
    data_path: Optional[str] = None,
    sensitive_attr: str = 'race',
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict:
    """
    Load the COMPAS Recidivism Dataset.
    
    ProPublica's COMPAS dataset for predicting recidivism.
    
    Parameters
    ----------
    data_path : str, optional
        Path to local CSV. If None, downloads from GitHub.
    sensitive_attr : str
        Sensitive attribute: 'race' or 'sex'
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
        
    Returns
    -------
    Dict with train/test data and metadata
    """
    if data_path is None:
        url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(data_path)
    
    # Filter as per ProPublica methodology
    df = df[(df['days_b_screening_arrest'] <= 30) & 
            (df['days_b_screening_arrest'] >= -30)]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']
    
    # Select relevant features
    features = ['age', 'c_charge_degree', 'priors_count', 'days_b_screening_arrest',
                'decile_score', 'score_text', 'sex', 'race', 'juv_fel_count', 
                'juv_misd_count', 'juv_other_count']
    
    df = df[features + ['two_year_recid']].dropna()
    
    # Encode categorical variables
    df['sex_binary'] = (df['sex'] == 'Male').astype(int)
    # For race, focus on African-American vs others (as in ProPublica analysis)
    df['race_binary'] = (df['race'] == 'African-American').astype(int)
    
    categorical_cols = ['c_charge_degree', 'score_text']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Select sensitive attribute
    if sensitive_attr == 'race':
        sensitive_col = 'race_binary'
    elif sensitive_attr == 'sex':
        sensitive_col = 'sex_binary'
    else:
        raise ValueError(f"Unknown sensitive attribute: {sensitive_attr}")
    
    # Prepare features
    feature_cols = ['age', 'c_charge_degree', 'priors_count', 'days_b_screening_arrest',
                   'juv_fel_count', 'juv_misd_count', 'juv_other_count']
    
    X = df[feature_cols].values
    y = df['two_year_recid'].values
    sensitive = df[sensitive_col].values
    
    # Split
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'sensitive_train': sens_train,
        'sensitive_test': sens_test,
        'sensitive_attr': sensitive_attr,
        'feature_names': feature_cols,
        'scaler': scaler,
        'encoders': encoders,
        'metadata': {
            'name': 'COMPAS',
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'task': 'Recidivism prediction',
            'positive_label': 'Recidivism within 2 years',
            'sensitive_groups': {0: 'Other races' if sensitive_attr == 'race' else 'Female',
                               1: 'African-American' if sensitive_attr == 'race' else 'Male'}
        }
    }


def load_bank_marketing(
    data_path: Optional[str] = None,
    sensitive_attr: str = 'age',
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict:
    """
    Load the Bank Marketing Dataset.
    
    UCI Bank Marketing dataset for predicting term deposit subscription.
    
    Parameters
    ----------
    data_path : str, optional
        Path to local CSV. If None, downloads from UCI.
    sensitive_attr : str
        Sensitive attribute: 'age' or 'marital'
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
        
    Returns
    -------
    Dict with train/test data and metadata
    """
    if data_path is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
        import urllib.request
        import zipfile
        import io
        
        try:
            with urllib.request.urlopen(url) as response:
                with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                    # Read the full dataset
                    with z.open('bank-additional/bank-additional-full.csv') as f:
                        df = pd.read_csv(f, sep=';')
        except Exception as e:
            # Fallback: create synthetic-like data structure
            print(f"Could not download Bank Marketing dataset: {e}")
            print("Using alternative URL...")
            # Try alternative approach
            alt_url = "https://raw.githubusercontent.com/selva86/datasets/master/bank-additional-full.csv"
            df = pd.read_csv(alt_url, sep=';')
    else:
        df = pd.read_csv(data_path, sep=';')
    
    # Create binary age (young < 35, old >= 35)
    df['age_binary'] = (df['age'] >= 35).astype(int)
    
    # Create binary marital status (single vs not single)
    df['marital_binary'] = (df['marital'] == 'single').astype(int)
    
    # Target encoding
    df['y'] = (df['y'] == 'yes').astype(int)
    
    # Encode categorical variables
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                       'contact', 'month', 'day_of_week', 'poutcome']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Select sensitive attribute
    if sensitive_attr == 'age':
        sensitive_col = 'age_binary'
    elif sensitive_attr == 'marital':
        sensitive_col = 'marital_binary'
    else:
        raise ValueError(f"Unknown sensitive attribute: {sensitive_attr}")
    
    # Feature columns (exclude target and sensitive derivatives)
    exclude_cols = ['y', 'age_binary', 'marital_binary', 'duration']  # duration leads to data leakage
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['y'].values
    sensitive = df[sensitive_col].values
    
    # Split
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'sensitive_train': sens_train,
        'sensitive_test': sens_test,
        'sensitive_attr': sensitive_attr,
        'feature_names': feature_cols,
        'scaler': scaler,
        'encoders': encoders,
        'metadata': {
            'name': 'Bank Marketing',
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'task': 'Term deposit subscription prediction',
            'positive_label': 'Subscribed',
            'sensitive_groups': {0: 'Young (<35)' if sensitive_attr == 'age' else 'Not single',
                               1: 'Old (>=35)' if sensitive_attr == 'age' else 'Single'}
        }
    }


def load_law_school(
    data_path: Optional[str] = None,
    sensitive_attr: str = 'race',
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict:
    """
    Load the Law School Admissions Dataset.
    
    LSAC dataset for predicting bar exam passage.
    
    Parameters
    ----------
    data_path : str, optional
        Path to local CSV.
    sensitive_attr : str
        Sensitive attribute: 'race' or 'sex'
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
        
    Returns
    -------
    Dict with train/test data and metadata
    """
    if data_path is None:
        # Try to load from common fairness datasets repository
        url = "https://raw.githubusercontent.com/tailequy/fairness_dataset/main/experiments/data/law_school_clean.csv"
        try:
            df = pd.read_csv(url)
        except Exception:
            # Create synthetic law school-like dataset
            print("Creating synthetic Law School dataset for demonstration...")
            np.random.seed(random_state)
            n_samples = 21000
            
            df = pd.DataFrame({
                'decile1b': np.random.randint(1, 11, n_samples),  # LSAT decile
                'decile3': np.random.randint(1, 11, n_samples),   # GPA decile
                'lsat': np.random.normal(35, 8, n_samples).clip(10, 48),
                'ugpa': np.random.normal(3.2, 0.5, n_samples).clip(1.5, 4.0),
                'zfygpa': np.random.normal(0, 1, n_samples),
                'zgpa': np.random.normal(0, 1, n_samples),
                'fulltime': np.random.binomial(1, 0.85, n_samples),
                'fam_inc': np.random.randint(1, 6, n_samples),
                'sex': np.random.binomial(1, 0.45, n_samples),
                'race': np.random.binomial(1, 0.15, n_samples),  # 1 = minority
            })
            
            # Generate target based on realistic correlations
            prob = 1 / (1 + np.exp(-(0.1 * df['lsat'] + 0.5 * df['ugpa'] - 5 + 
                                     np.random.normal(0, 0.5, n_samples))))
            df['pass_bar'] = (np.random.random(n_samples) < prob).astype(int)
    else:
        df = pd.read_csv(data_path)
    
    # Handle column names (different sources may have different conventions)
    col_mapping = {
        'male': 'sex',
        'white': 'race',
        'bar': 'pass_bar',
        'pass': 'pass_bar'
    }
    df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns}, inplace=True)
    
    # Ensure binary encoding
    if 'race' in df.columns:
        # 1 = White/majority, 0 = minority (or vice versa depending on source)
        if df['race'].dtype == object:
            df['race'] = (df['race'].str.lower().isin(['white', 'caucasian'])).astype(int)
    
    if 'sex' in df.columns:
        if df['sex'].dtype == object:
            df['sex'] = (df['sex'].str.lower() == 'male').astype(int)
    
    # Select target column
    target_col = 'pass_bar' if 'pass_bar' in df.columns else 'bar'
    if target_col not in df.columns:
        raise ValueError("Could not find target column in Law School dataset")
    
    # Select sensitive attribute
    if sensitive_attr == 'race':
        sensitive_col = 'race'
    elif sensitive_attr == 'sex':
        sensitive_col = 'sex'
    else:
        raise ValueError(f"Unknown sensitive attribute: {sensitive_attr}")
    
    # Feature columns
    exclude_cols = [target_col, 'race', 'sex']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    
    # Drop rows with missing values
    df = df.dropna(subset=feature_cols + [target_col, sensitive_col])
    
    X = df[feature_cols].values
    y = df[target_col].values.astype(int)
    sensitive = df[sensitive_col].values.astype(int)
    
    # Split
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'sensitive_train': sens_train,
        'sensitive_test': sens_test,
        'sensitive_attr': sensitive_attr,
        'feature_names': feature_cols,
        'scaler': scaler,
        'metadata': {
            'name': 'Law School',
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'task': 'Bar exam passage prediction',
            'positive_label': 'Passed bar exam',
            'sensitive_groups': {0: 'Minority' if sensitive_attr == 'race' else 'Female',
                               1: 'White' if sensitive_attr == 'race' else 'Male'}
        }
    }


def load_adult_census(
    data_path: Optional[str] = None,
    sensitive_attr: str = 'sex',
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict:
    """
    Load the Adult Census Income Dataset.
    
    UCI Adult dataset for predicting income >50K.
    
    Parameters
    ----------
    data_path : str, optional
        Path to local CSV. If None, downloads from UCI.
    sensitive_attr : str
        Sensitive attribute: 'sex' or 'race'
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
        
    Returns
    -------
    Dict with train/test data and metadata
    """
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    if data_path is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        df = pd.read_csv(url, names=columns, sep=',\\s*', engine='python', na_values='?')
    else:
        df = pd.read_csv(data_path, names=columns if 'income' not in open(data_path).readline() else None)
    
    # Handle missing values
    df = df.dropna()
    
    # Encode target
    df['income'] = (df['income'].str.strip() == '>50K').astype(int)
    
    # Binary encoding for sensitive attributes
    df['sex_binary'] = (df['sex'].str.strip() == 'Male').astype(int)
    df['race_binary'] = (df['race'].str.strip() == 'White').astype(int)
    
    # Encode categorical variables
    categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                       'relationship', 'race', 'sex', 'native_country']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Select sensitive attribute
    if sensitive_attr == 'sex':
        sensitive_col = 'sex_binary'
    elif sensitive_attr == 'race':
        sensitive_col = 'race_binary'
    else:
        raise ValueError(f"Unknown sensitive attribute: {sensitive_attr}")
    
    # Feature columns
    exclude_cols = ['income', 'sex_binary', 'race_binary']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].values
    y = df['income'].values
    sensitive = df[sensitive_col].values
    
    # Split
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'sensitive_train': sens_train,
        'sensitive_test': sens_test,
        'sensitive_attr': sensitive_attr,
        'feature_names': feature_cols,
        'scaler': scaler,
        'encoders': encoders,
        'metadata': {
            'name': 'Adult Census',
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'task': 'Income prediction',
            'positive_label': 'Income >50K',
            'sensitive_groups': {0: 'Female' if sensitive_attr == 'sex' else 'Non-white',
                               1: 'Male' if sensitive_attr == 'sex' else 'White'}
        }
    }


# Dataset registry for easy access
DATASET_LOADERS = {
    'german_credit': load_german_credit,
    'compas': load_compas,
    'bank_marketing': load_bank_marketing,
    'law_school': load_law_school,
    'adult_census': load_adult_census
}


def load_dataset(name: str, **kwargs) -> Dict:
    """
    Load a dataset by name.
    
    Parameters
    ----------
    name : str
        Dataset name: 'german_credit', 'compas', 'bank_marketing', 'law_school', 'adult_census'
    **kwargs
        Additional arguments passed to the loader
        
    Returns
    -------
    Dict with train/test data and metadata
    """
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")
    
    return DATASET_LOADERS[name](**kwargs)


def get_dataset_summary(data: Dict) -> pd.DataFrame:
    """
    Create a summary DataFrame for a loaded dataset.
    
    Parameters
    ----------
    data : Dict
        Output from a dataset loader
        
    Returns
    -------
    pd.DataFrame with dataset statistics
    """
    metadata = data['metadata']
    
    # Compute statistics
    train_pos_rate = np.mean(data['y_train'])
    test_pos_rate = np.mean(data['y_test'])
    
    train_sens_dist = np.mean(data['sensitive_train'])
    test_sens_dist = np.mean(data['sensitive_test'])
    
    # Group-wise positive rates
    sens_0_train = data['y_train'][data['sensitive_train'] == 0]
    sens_1_train = data['y_train'][data['sensitive_train'] == 1]
    
    summary = {
        'Dataset': metadata['name'],
        'Task': metadata['task'],
        'N Train': len(data['y_train']),
        'N Test': len(data['y_test']),
        'N Features': len(data['feature_names']),
        'Positive Rate (Train)': f"{train_pos_rate:.2%}",
        'Sensitive Attr': data['sensitive_attr'],
        f"Group 0 Rate": f"{1 - train_sens_dist:.2%}",
        f"Group 1 Rate": f"{train_sens_dist:.2%}",
        'Group 0 Pos Rate': f"{np.mean(sens_0_train):.2%}",
        'Group 1 Pos Rate': f"{np.mean(sens_1_train):.2%}",
        'Disparate Impact': f"{np.mean(sens_0_train) / np.mean(sens_1_train):.3f}" if np.mean(sens_1_train) > 0 else "N/A"
    }
    
    return pd.DataFrame([summary])
