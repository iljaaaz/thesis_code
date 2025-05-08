import os
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm


def extract_movement_features(df):
    """Enhanced with value clipping and NaN handling"""
    # Time differentials with clipping
    df['dt'] = df['datetime'].diff().dt.total_seconds().clip(lower=0.01, upper=10)
    
    # Displacement components with clipping
    df['dx'] = df['x'].diff().fillna(0).clip(lower=-1000, upper=1000)
    df['dy'] = df['y'].diff().fillna(0).clip(lower=-1000, upper=1000)
    df['distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # Component velocitie
    df['vx'] = df['dx'] / df['dt']
    df['vy'] = df['dy'] / df['dt']
    df['vt'] = np.sqrt(df['vx']**2 + df['vy']**2)
    
    # Replace infinite velocities
    for col in ['vx', 'vy', 'vt']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    # Component accelerations
    df['ax'] = df['vx'].diff() / df['dt']
    df['ay'] = df['vy'].diff() / df['dt']
    df['at'] = df['vt'].diff() / df['dt']
    
    # Jitter calculation with clipping
    df['jitter_x'] = df['dx'].diff().abs() / df['dt'].clip(lower=0.01)
    df['jitter_y'] = df['dy'].diff().abs() / df['dt'].clip(lower=0.01)
    
    return df

def extract_session_features(session_df):
    """Feature extraction focusing on movement kinematics"""
    # Preprocessing
    session_df['datetime'] = pd.to_datetime(session_df['client_timestamp'], unit='s')
    session_df = session_df.sort_values('datetime')
    
    # Filter only movement events
    movement_df = session_df[session_df['button'].isna()].copy()
    if len(movement_df) < 20:  # Minimum movement events threshold
        raise ValueError("Insufficient movement data")
    
    # Extract kinematic features
    movement_df = extract_movement_features(movement_df)
    
    # Calculate features
    features = {
        # Velocity features
        'vx_mean': movement_df['vx'].mean(),
        'vx_std': movement_df['vx'].std(),
        'vy_mean': movement_df['vy'].mean(),
        'vy_std': movement_df['vy'].std(),
        'vt_mean': movement_df['vt'].mean(),
        'vt_std': movement_df['vt'].std(),
        
        # Acceleration features
        'ax_mean': movement_df['ax'].mean(),
        'ax_std': movement_df['ax'].std(),
        'ay_mean': movement_df['ay'].mean(),
        'ay_std': movement_df['ay'].std(),
        'at_mean': movement_df['at'].mean(),
        'at_std': movement_df['at'].std(),
        
        # Jitter metrics
        'jitter_x_mean': movement_df['jitter_x'].mean(),
        'jitter_y_mean': movement_df['jitter_y'].mean(),
        'jitter_ratio': movement_df['jitter_y'].mean() / (movement_df['jitter_x'].mean() + 1e-10),
        
        # Movement quality
        'straightness': (movement_df['distance'].sum() / 
                        (np.sqrt((movement_df['x'].iloc[-1] - movement_df['x'].iloc[0])**2 + 
                         (movement_df['y'].iloc[-1] - movement_df['y'].iloc[0])**2 + 1e-10)))
    }
    
    return features

def process_user(user_dir, user_id):
    """Process a single user's sessions"""
        
    training_dir = os.path.join(user_dir, "training")
    if not os.path.exists(training_dir):
        return None
        
    session_files = [f for f in os.listdir(training_dir) 
                   if f.startswith('session_') and f.endswith('.csv')]
    
    user_features = []
    for session_file in session_files:
        try:
            session_path = os.path.join(training_dir, session_file)
            session_df = pd.read_csv(session_path)
            
            features = extract_session_features(session_df)
            features['session_id'] = session_file
            user_features.append(features)
            
        except Exception as e:
            continue
            
    if user_features:
        user_df = pd.DataFrame(user_features)
        user_df['user_id'] = user_id
        return user_df
    return None

def process_all_users(dataset_path="boun-mouse-dynamics-dataset/users"):
    """Process all non-excluded users"""
    user_dirs = [d for d in os.listdir(dataset_path) 
                if d.startswith('user') and 
                int(d[4:])]
    
    all_features = []
    for user_dir in tqdm(sorted(user_dirs), desc="Processing Users"):
        user_id = int(user_dir[4:])
        user_path = os.path.join(dataset_path, user_dir)
        
        user_df = process_user(user_path, user_id)
        if user_df is not None:
            all_features.append(user_df)
    
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    return pd.DataFrame()


# Execute and save results
if __name__ == "__main__":
    features_df = process_all_users()
    
    if not features_df.empty:
        # Save with timestamp
        output_file = f"mouse_kinematics_features_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        features_df.to_csv(output_file, index=False)
        print(f"Successfully processed {features_df['user_id'].nunique()} users")
        print(f"Results saved to {output_file}")
        print("\nFeature summary:\n", features_df.describe())
    else:
        print("No valid data processed")
