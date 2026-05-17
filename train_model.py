import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

def normalize_hand_coordinates(raw_coords):    
    if sum(raw_coords)==0.0:
        return list(raw_coords) 

    wrist_x=raw_coords[0]
    wrist_y=raw_coords[1]

    shifted_coords=[]
    for crd in range(0, len(raw_coords), 2):
        shifted_coords.append(raw_coords[crd] - wrist_x)     
        shifted_coords.append(raw_coords[crd+1] - wrist_y)   

    max_value=max(max(abs(val) for val in shifted_coords), 1e-6)
    return [val / max_value for val in shifted_coords]

data=pd.read_csv('hand_data.csv', header=None)

print(f"Rows prior cleaning: {len(data)}")
data=data.dropna() 
print(f"Rows after cleaning: {len(data)}")

X_raw=data.iloc[:,1:].values
Y = data.iloc[:,0].values

print("Transforming screen space coordinates to wrist space coordinates.")
X_normalized=[]
for row in X_raw:
    row_normalized=[]
    
    for frame_offset in range(0, len(row), 84):
        frame_chunk=row[frame_offset : frame_offset + 84]
        
        right_hand=list(frame_chunk[0:42])
        left_hand=list(frame_chunk[42:84])
        
        norm_r=normalize_hand_coordinates(right_hand)
        norm_l=normalize_hand_coordinates(left_hand)
        
        row_normalized.extend(norm_r + norm_l)
        
    X_normalized.append(row_normalized)

X=np.array(X_normalized)


X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, stratify=Y)

print("Training the model...!")
model=RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

predictor=model.predict(X_test)
score=accuracy_score(Y_test, predictor)

print(f"Training Complete! Accuracy: {score * 100:.2f}%")


with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Brain saved as 'model.p'")