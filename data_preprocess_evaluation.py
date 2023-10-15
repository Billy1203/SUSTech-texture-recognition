import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import random
from scipy.stats import kurtosis, skew
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame, impute
from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# Data Preprocessing
folder_path = "./dataset/dataset4/raw_data"
data = {}
for i in range(1, 21):
    filename = f"{i}.txt"
    file_path = os.path.join(folder_path, filename)
    with open(file_path, "r") as file:
        lines = file.readlines()
        values = np.array([float(line.strip()) for line in lines])
        data[i] = values

# Feature Extraction
df = pd.DataFrame()
directory = "./dataset/dataset4"

# Loading signals and ensuring each has a length of 20
for i in range(1, 21):
    file_name = f"dt{i}.npy"
    file_path = os.path.join(directory, file_name)
    signal_array = np.load(file_path)
    selected_signals = signal_array[:20] if len(signal_array) >= 20 else np.pad(signal_array, (0, 20-len(signal_array)), 'edge')
    df_temp = pd.DataFrame({"Signal": selected_signals.tolist(), "Label": i})
    df = pd.concat([df, df_temp], ignore_index=True)

df['Signal'] = df['Signal'].apply(lambda list_: [int(x) for x in list_])

# Use tsfresh for feature extraction
settings = EfficientFCParameters()
settings_minimal = MinimalFCParameters()
X = np.load("./dataset4/dataset_X.npy", allow_pickle=True)
y = np.load("./dataset4/dataset_y.npy", allow_pickle=True)
df = pd.DataFrame({'Signal': X, 'Label': y})
df['Label'] = df['Label'].astype(int)
df.sort_values("Label", inplace=True)

df_tsfresh = df['Signal'].apply(pd.Series).stack().reset_index()
df_tsfresh.columns = ['id', 'time', 'value']

# Extract features using Efficient and Minimal settings
extracted_features = extract_features(df_tsfresh, column_id='id', column_sort='time', default_fc_parameters=settings)
extracted_features_minimal = extract_features(df_tsfresh, column_id='id', column_sort='time', default_fc_parameters=settings_minimal)

# Filling empty values
extracted_features.fillna(0, inplace=True)
extracted_features_minimal.fillna(0, inplace=True)

# Data Augmentation by adding random samples
for label, group in extracted_features.groupby('Label'):
    extracted_features = extracted_features.append(group.sample(n=1, replace=True), ignore_index=True)
for label, group in extracted_features_minimal.groupby('Label'):
    extracted_features_minimal = extracted_features_minimal.append(group.sample(n=1, replace=True), ignore_index=True)

train_df, test_df = train_test_split(extracted_features, test_size=0.4, stratify=extracted_features['Label'], random_state=42)

X_train = train_df.drop(['Label', 'Signal'], axis=1).values
y_train = train_df['Label'].values
X_test = test_df.drop(['Label', 'Signal'], axis=1).values
y_test = test_df['Label'].values

# Data Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification and Evaluation
classifiers = [SVC(), LogisticRegression(max_iter=1000), RandomForestClassifier(), DecisionTreeClassifier(),
               KNeighborsClassifier(), GaussianNB()]
classifier_names = ["SVM", "Logistic Regression", "Random Forest", "Decision Tree", "K Nearest Neighbors",
                    "Gaussian Naive Bayes"]

for clf, clf_name in zip(classifiers, classifier_names):
    start = time.time()
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm / cm.sum(axis=1).reshape(-1, 1) * 100

    # Confusion Matrix Visualization
    plt.figure(figsize=(15, 13))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Percentages) - {clf_name}')
    plt.show()
