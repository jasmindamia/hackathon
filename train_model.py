import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# 1. Load the data
# If your CSV doesn't have headers, we tell pandas not to look for them
data = pd.read_csv('hand_data.csv', header=None)

# 2. Split into Features (X) and Labels (y)
# X = all columns except the first one (the coordinates)
# y = the first column (the word/label)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 3. Split into Training and Testing sets
# We use 80% of data to teach the AI and 20% to test it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 4. Initialize and Train the "Brain" (Random Forest)
print("Training the brain... this might take a few seconds.")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 5. Check Accuracy
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)

print(f"Training Complete! Accuracy: {score * 100:.2f}%")

# 6. Save the model to a file so Level 4 can use it
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Brain saved as 'model.p'. You are ready for Level 4!")