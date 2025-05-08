import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

def prepare_data(features_path, genuine_user_id):
    """Mark one user as genuine and others as imposters"""
    df = pd.read_csv(features_path)
    df['label'] = np.where(df['user_id'] == genuine_user_id, 1, 0)
    return df

def calculate_metrics(y_true, y_proba):
    """Calculate EER, AU-ROC with interpolation"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # Calculate EER
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    auc = roc_auc_score(y_true, y_proba)
    
    return eer, auc

def evaluate_model(X_train, X_test, y_train, y_test, model):
    """Evaluate a single model instance"""
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:,1]
    return calculate_metrics(y_test, y_proba)

def run_user_experiments(features_path):
    """Evaluate all users and compute average performance"""
    # Get unique user IDs
    df = pd.read_csv(features_path)
    user_ids = df['user_id'].unique()
    
    # Model templates
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         solver='adam', random_state=42)
        )
    }
    
    # Store results
    results = {name: {'eers': [], 'aucs': []} for name in models.keys()}
    
    # Evaluate each user
    for user_id in tqdm(user_ids, desc="Evaluating Users"):
        data = prepare_data(features_path, user_id)
        feature_cols = [c for c in data.columns if c not in ['user_id', 'session_id', 'label']]
        X = data[feature_cols]
        y = data['label']
        
        # Split data (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Evaluate all models
        for name, model in models.items():
            eer, auc = evaluate_model(X_train, X_test, y_train, y_test, model)
            results[name]['eers'].append(eer)
            results[name]['aucs'].append(auc)
    
    # Compute averages
    avg_results = {}
    for name in models.keys():
        avg_results[name] = {
            'avg_eer': np.mean(results[name]['eers']),
            'std_eer': np.std(results[name]['eers']),
            'avg_auc': np.mean(results[name]['aucs']),
            'std_auc': np.std(results[name]['aucs'])
        }
    
    return avg_results, results

def plot_avg_performance(avg_results):
    """Plot average model performance"""
    metrics = ['avg_eer', 'avg_auc']
    titles = ['Equal Error Rate (Lower Better)', 'AU-ROC (Higher Better)']
    
    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i+1)
        names = list(avg_results.keys())
        values = [avg_results[name][metric] for name in names]
        errors = [avg_results[name][f'std_{metric[4:]}'] for name in names]
        
        plt.bar(names, values, yerr=errors, capsize=5)
        plt.title(titles[i])
        plt.ylabel(metric.split('_')[-1].upper())
        if metric == 'avg_eer':
            plt.ylim(0, 0.5)  # EER typically 0-0.5
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    features_path = "mouse_kinematics_features_20250412.csv"
    
    # Run evaluation
    avg_results, all_results = run_user_experiments(features_path)
    
    # Print results
    print("\n=== Average Performance Across Users ===")
    for name, metrics in avg_results.items():
        print(f"\n{name}:")
        print(f"  EER: {metrics['avg_eer']:.4f} ± {metrics['std_eer']:.4f}")
        print(f"  AU-ROC: {metrics['avg_auc']:.4f} ± {metrics['std_auc']:.4f}")
    
    # Plot comparison
    plot_avg_performance(avg_results)
