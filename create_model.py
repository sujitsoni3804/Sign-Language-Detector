import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading dataset...")
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except FileNotFoundError:
    print("Error: data.pickle not found. Run create_dataset.py first.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

raw_data = data_dict['data']
labels = np.array(data_dict['labels'])

# Ensure all data arrays have the same length
max_len = max(len(arr) for arr in raw_data)
print(f"Feature length: {max_len}")

# Pad shorter arrays with zeros to make uniform length
padded_data = []
for arr in raw_data:
    if len(arr) < max_len:
        padded_arr = arr + [0] * (max_len - len(arr))
        padded_data.append(padded_arr)
    else:
        padded_data.append(arr[:max_len])  # Truncate if longer than expected

# Convert to numpy array
data = np.array(padded_data)

# Print dataset info
unique_labels = np.unique(labels)
print(f"Dataset shape: {data.shape}")
print(f"Number of classes: {len(unique_labels)}")
print("Labels distribution:")
for label in sorted(unique_labels, key=lambda x: int(x)):
    count = np.sum(labels == label)
    print(f"  Class {label}: {count} samples")

# Split into training and testing sets
try:
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
except ValueError as e:
    print(f"Error splitting data: {e}")
    print("Make sure you have samples for all classes you want to recognize.")
    exit()

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print("Training model...")

# Use Grid Search for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")

# Evaluate the model
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create confusion matrix visualization
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the visualization
if not os.path.exists('./outputs'):
    os.makedirs('./outputs')
plt.savefig('./outputs/confusion_matrix.png')
print("Confusion matrix saved to ./outputs/confusion_matrix.png")

# Save the best model and scaler
model_data = {
    'model': best_model,
    'scaler': scaler,
    'feature_len': max_len,
    'classes': sorted([int(l) for l in unique_labels])
}

with open('model.p', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved as model.p")
print("Now run main.py to test the sign language recognition system.")