import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Sample dataset
data = {
    'hours_studied': [1,2,3,4,5,6,7,8,9,10],
    'attendance': [60,65,70,75,80,85,90,92,95,98],
    'previous_marks': [50,55,60,65,70,75,80,85,90,95],
    'final_marks': [52,57,63,68,74,79,85,88,93,97]
}

df = pd.DataFrame(data)

X = df[['hours_studied', 'attendance', 'previous_marks']]
y = df['final_marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, 'model/student_model.pkl')

print("Model trained and saved successfully!")