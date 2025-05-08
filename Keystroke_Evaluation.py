import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from scipy.sparse import issparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import FunctionTransformer

# Configuration
DATASET_PATH = "BB-MAS_DATASET"
TEST_USER_ID = 57  # Selected user 
RANDOM_STATE = 3
TEST_SIZE = 0.3
NUM_IMPOSTORS = 7
MIN_SAMPLES_PER_USER = 250  # Minimum samples required per user
SEQUENCE_LENGTH = 12  # For LSTM

def calculate_eer(far, frr):
    """Calculate Equal Error Rate (EER)"""
    # Find the threshold where FAR and FRR are closest
    eer_threshold = np.nanargmin(np.abs(far - frr))
    eer = (far[eer_threshold] + frr[eer_threshold]) / 2
    return eer, eer_threshold

def load_raw_keystrokes(user_id):
    # This functions loads raw keystroke data for a user
    """Load raw keystroke data for a specific user"""
    print(f"\n[1/5] Loading raw keystrokes for user {user_id}...")
    file_path = os.path.join(DATASET_PATH, str(user_id), f"{user_id}_Desktop_Keyboard.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path, names=["EID", "key", "direction", "time"], header=0)
    df['time'] = pd.to_datetime(df['time'])
    print(f"Loaded {len(df)} raw keystroke events")
    return df

def extract_features_from_raw(df):
    # This function extracts keystroke features from raw data
    """Extract dwell time and all four flight times from raw keystroke data"""
    print("[2/5] Extracting features from raw data...")
    features = []
    
    # Verify and clean input data
    if df.empty:
        raise ValueError("Empty dataframe provided")
    
    required_columns = ['EID', 'key', 'direction', 'time']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert time and sort by EID
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('EID')
    
    # Separate presses and releases
    presses = df[df['direction'] == 0]
    releases = df[df['direction'] == 1]
    
    # Create press-release pairs
    paired_events = []
    release_idx = 0
    
    for _, press in presses.iterrows():
        # Find the corresponding release (next release of same key)
        while release_idx < len(releases):
            release = releases.iloc[release_idx]
            release_idx += 1
            if release['key'] == press['key']:
                paired_events.append((press, release))
                break
    
    # Calculate dwell times
    print(" - Calculating dwell times...")
    for press, release in paired_events:
        dwell = (release['time'] - press['time']).total_seconds() * 1000
        features.append({
            'EID': press['EID'],
            'key': press['key'],
            'type': 'dwell',
            'time': dwell,
            'direction': 'press',
            'key1': press['key'],
            'key2': None
        })
    
    # Calculate flight times between consecutive presses
    print(" - Calculating flight times...")
    for i in range(len(paired_events)-1):
        curr_press, curr_release = paired_events[i]
        next_press, next_release = paired_events[i+1]
        
        # Calculate all four flight times
        flight1 = (next_press['time'] - curr_release['time']).total_seconds() * 1000
        flight2 = (next_release['time'] - curr_release['time']).total_seconds() * 1000
        flight3 = (next_press['time'] - curr_press['time']).total_seconds() * 1000
        flight4 = (next_release['time'] - curr_press['time']).total_seconds() * 1000
        
        # Append flight time features
        flight_data = {
            'flight1': flight1,
            'flight2': flight2,
            'flight3': flight3,
            'flight4': flight4
        }
        
        for flight_type, time in flight_data.items():
            features.append({
                'EID': curr_press['EID'],
                'key': None,
                'type': flight_type,
                'time': time,
                'direction': 'flight',
                'key1': curr_press['key'],
                'key2': next_press['key']
            })
    
    if not features:
        raise ValueError("No features extracted - check input data structure")
    
    features_df = pd.DataFrame(features)
    print(f"Successfully extracted {len(features_df)} features")
    print("Feature type counts:")
    print(features_df['type'].value_counts())
    
    return features_df

def create_balanced_dataset(genuine_user_id, num_impostors=NUM_IMPOSTORS):
    # This function creates the final dataset
    """Create balanced dataset with genuine and impostor samples"""
    print(f"\n[3/5] Creating balanced dataset for user {genuine_user_id}...")
    
    # Genuine user features - collect multiple sessions
    print(" - Processing genuine user...")
    genuine_features = []
    raw_df = load_raw_keystrokes(genuine_user_id)
    genuine_raw_features = extract_features_from_raw(raw_df)
    
    # Combine genuine features
    genuine_features = pd.concat([genuine_raw_features])
    genuine_features['label'] = 1  # 1 = genuine, 0 = impostor
    print(f" - Genuine samples collected: {len(genuine_features)}")
    
    # Impostor features (from other users)
    print(f" - Processing {num_impostors} impostors...")
    impostor_features = []
    all_users = [int(f) for f in os.listdir(DATASET_PATH) if f.isdigit() and int(f) != genuine_user_id]
    impostor_users = np.random.choice(all_users, min(num_impostors, len(all_users)), replace=False)
    
    for i, user_id in enumerate(impostor_users, 1):
        try:
            print(f"   - Impostor {i}/{len(impostor_users)}: user {user_id}")
            raw_df = load_raw_keystrokes(user_id)
            impostor_raw = extract_features_from_raw(raw_df)
            impostor_combined = pd.concat([impostor_raw])
            impostor_combined['label'] = 0  # 0 = impostor, 1 = genuine
            impostor_features.append(impostor_combined)
        except Exception as e:
            print(f"     Error processing user {user_id}: {str(e)}")
            continue
    
    if impostor_features:
        impostor_features = pd.concat(impostor_features)
        print(f" - Impostor samples collected: {len(impostor_features)}")
    else:
        print(" - No impostor samples collected!")
        impostor_features = pd.DataFrame()
    
    # Ensure we have enough samples
    if len(genuine_features) < MIN_SAMPLES_PER_USER or len(impostor_features) < MIN_SAMPLES_PER_USER:
        raise ValueError(f"Not enough samples (min {MIN_SAMPLES_PER_USER} required). Genuine: {len(genuine_features)}, Impostor: {len(impostor_features)}")
    
    # Balance classes by taking min samples
    min_samples = min(len(genuine_features), len(impostor_features), MIN_SAMPLES_PER_USER)
    print(f"\nBalancing classes to {min_samples} samples each...")
    
    df_genuine = genuine_features.sample(min_samples, random_state=RANDOM_STATE)
    df_impostor = impostor_features.sample(min_samples, random_state=RANDOM_STATE)
    
    dataset = pd.concat([df_genuine, df_impostor]).sample(frac=1, random_state=RANDOM_STATE)
    print(f"\nFinal balanced dataset: {len(dataset)} samples ({min_samples} genuine, {min_samples} impostor)")
    return dataset

def create_feature_pipeline():
    """Create preprocessing pipeline for features"""
    # Numerical features (times)
    numeric_features = ['time']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical features (keys)
    categorical_features = ['key', 'key1', 'key2']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def create_mlp_model(input_shape):
    """Create MLP model architecture"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def create_lstm_model(input_shape):
    """Create LSTM model architecture"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def reshape_for_dl(X):
    """Reshape data for deep learning models"""
    return X.reshape((X.shape[0], 1, X.shape[1]))

def train_models(X_train, y_train):
    """Train optimized machine learning models"""
    print("\n[4/5] Training models...")
    
    # Create feature pipeline
    preprocessor = create_feature_pipeline()
    
    # Preprocess the data to get feature dimension
    X_temp = preprocessor.fit_transform(X_train)
    input_shape = (X_temp.shape[1],)
    lstm_input_shape = (1, X_temp.shape[1])  # For sequence models
    
    models = {
        'Logistic Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C= 0.1,
                penalty='l2',
                class_weight={0: 1, 1:3},
                max_iter=1000,
                solver='liblinear',
                random_state=RANDOM_STATE
            ))
        ]),
        'KNN': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='manhattan'
            ))
        ]),
        'Naive Bayes': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', GaussianNB())
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                class_weight={0: 1, 1:3},
                random_state=RANDOM_STATE,
                min_samples_split=5,
                min_samples_leaf=2
            ))
        ]),
        'SVM': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                class_weight={0: 1, 1:3},
                max_iter=1000,
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),
        'MLP': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KerasClassifier(
                class_weight={0: 1, 1:5},
                build_fn=lambda: create_mlp_model(input_shape),
                epochs=100,
                batch_size=32,
                verbose=0
            ))
        ]),
        'LSTM': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('reshape', FunctionTransformer(reshape_for_dl, validate=False)),
            ('classifier', KerasClassifier(
                class_weight={0: 1, 1:5},
                build_fn=lambda: create_lstm_model(lstm_input_shape),
                epochs=100,
                batch_size=32,
                verbose=0
            ))
        ])
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f" - Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
        except Exception as e:
            print(f"   Error training {name}: {str(e)}")
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluation with detailed EER and AUROC analysis"""
    print("\n[5/5] Evaluating models...")
    
    results = []
    thresholds = np.linspace(0, 1, 101)  # 100 threshold points
    
    plt.figure(figsize=(15, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(models)))
    
    # Prepare subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Store evaluation metrics
    evaluation_results = []
    
    for (name, model), color in zip(models.items(), colors):
        try:
            print(f"\nEvaluating {name}...")
            
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            
            # Calculate confusion matrices across thresholds
            far = []
            frr = []
            acc = []
            
            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                
                far.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
                frr.append(fn / (fn + tp) if (fn + tp) > 0 else 0)
                acc.append((tp + tn) / (tp + tn + fp + fn))
            
            # Calculate EER
            eer_idx = np.nanargmin(np.abs(np.array(far) - np.array(frr)))
            eer = (far[eer_idx] + frr[eer_idx]) / 2
            eer_thresh = thresholds[eer_idx]
            
            # Calculate ROC metrics
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            # Store results
            evaluation_results.append({
                'Model': name,
                'EER': eer,
                'EER_Threshold': eer_thresh,
                'FAR_at_EER': far[eer_idx],
                'FRR_at_EER': frr[eer_idx],
                'Max_Accuracy': max(acc),
                'Best_Threshold': thresholds[np.argmax(acc)],
                'AUROC': roc_auc
            })
            
            # Print detailed metrics
            print(f"=== {name} Performance ===")
            print(f"AUROC: {roc_auc:.4f}")
            print(f"EER: {eer:.4f} (Threshold: {eer_thresh:.4f})")
            print(f"FAR at EER: {far[eer_idx]:.4f}")
            print(f"FRR at EER: {frr[eer_idx]:.4f}")
            print(f"Max Accuracy: {max(acc):.4f} at threshold {thresholds[np.argmax(acc)]:.4f}")
            
            # Plot FAR-FRR curves
            ax1.plot(thresholds, far, label=f'{name} FAR', color=color, linestyle='--')
            ax1.plot(thresholds, frr, label=f'{name} FRR', color=color, linestyle='-')
            ax1.scatter(eer_thresh, eer, color=color, marker='x')
            
            # Plot ROC curve
            ax2.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', color=color)
            
            # Plot accuracy vs threshold
            ax3.plot(thresholds, acc, label=name, color=color)
            
        except Exception as e:
            print(f"Error evaluating {name}: {str(e)}")
            continue
    
    # Format FAR-FRR plot
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('Rate')
    ax1.set_title('FAR and FRR vs Threshold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid()
    ax1.annotate('Lower is better', xy=(0.5, 0.9), xycoords='axes fraction')
    
    # Format ROC plot
    ax2.set_xlabel('False Positive Rate (FAR)')
    ax2.set_ylabel('True Positive Rate (1-FRR)')
    ax2.set_title('ROC Curve')
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid()
    
    # Format Accuracy plot
    ax3.set_xlabel('Decision Threshold')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy vs Threshold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid()
    ax3.annotate('Peak indicates optimal threshold', xy=(0.5, 0.9), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.show()
    
    # Create and display detailed results
    results_df = pd.DataFrame(evaluation_results)
    results_df = results_df.sort_values('EER')
    
    return results_df

def main():
    print("=== Evaluation of classifiers for Behavioral Biometrics based system ===")
    print(f"Starting analysis for user {TEST_USER_ID}")
    
    try:
        # 1. Create balanced dataset
        dataset = create_balanced_dataset(TEST_USER_ID)
        
        # Prepare features and labels
        X = dataset[['key', 'key1', 'key2', 'time', 'direction']]
        y = dataset['label']
        
        # 2. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        
        # 3. Train models
        models = train_models(X_train, y_train)
        
        # 4. Evaluate models only if we have successful models
        if models:
            results = evaluate_models(models, X_test, y_test)
        else:
            print("\nNo models were successfully trained")
        
        print("\n=== Analysis Complete ===")
    except ValueError as e:
        print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    main()
