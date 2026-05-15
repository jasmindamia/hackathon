import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('hand_data.csv', header=None)

print(f"Rows before cleaning: {len(data)}")
data = data.dropna() 
print(f"Rows after cleaning: {len(data)}")

#split into features(X) and labels(y)
# X;all columns except the first one (the coordinates)
# y;the first column (the word/label)
X=data.iloc[:,1:].values
y=data.iloc[:,0].values

#split into training and testing sets
#using 80% of data to teach the AI and 20% to test it
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,stratify=y)

#brain training
print("Training the brain... bsssttt!.")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

#check Accuracy
y_predict = model.predict(X_test)
score = accuracy_score(y_test, y_predict)

print(f"Training Complete! Accuracy: {score * 100:.2f}%")


with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Brain saved as 'model.p'. horray!!")