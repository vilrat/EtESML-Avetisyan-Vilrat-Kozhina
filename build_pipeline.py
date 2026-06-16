import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from category_encoders import HashingEncoder

NUM  = ['year', 'condition', 'odometer']
OHE  = ['body', 'transmission', 'state', 'color', 'interior']
HASH = ['make', 'model', 'trim', 'seller']
DROP = ['vin', 'saledate']           # ignored by the model

df = pd.read_csv('data/sample_2.csv').dropna(subset=['sellingprice']).copy()
X = df.drop(columns=['sellingprice'] + DROP)
y = df['sellingprice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

numeric = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
categorical = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore')),
])
hashing = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('hash', HashingEncoder(n_components=32)),
])

preprocessor = ColumnTransformer([
    ('num', numeric, NUM),
    ('cat', categorical, OHE),
    ('hash', hashing, HASH),
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)),
])

model.fit(X_train, y_train)

pred = model.predict(X_test)
print('RMSE: %.2f' % np.sqrt(mean_squared_error(y_test, pred)))
print('MAE : %.2f' % mean_absolute_error(y_test, pred))
print('R2  : %.4f' % r2_score(y_test, pred))

os.makedirs('backend', exist_ok=True)
joblib.dump(model, 'backend/best_model_pipeline.joblib', compress=3)
print('Saved backend/best_model_pipeline.joblib')
