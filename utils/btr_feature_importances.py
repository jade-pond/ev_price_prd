import streamlit as st
from xgboost import XGBRegressor
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 데이터 로드 함수
# @st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)


df = load_data("data/train.csv")


processed_df = df.dropna(axis=0).set_index('ID')

X = processed_df[[col for col in processed_df.columns if col != '배터리용량']]
categoric_cols = ['제조사', '모델', '차량상태','구동방식','사고이력']

for xcol in categoric_cols:
    le = LabelEncoder()
    X[xcol] = le.fit_transform(X[xcol])

y = processed_df['배터리용량']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model = XGBRegressor(random_state=42)

model.fit(X_train, y_train)

feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

with open('xgb_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('feature_importances.pkl', 'wb') as fi_file:
    pickle.dump(feature_importances, fi_file)