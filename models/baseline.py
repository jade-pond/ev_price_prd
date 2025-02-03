import streamlit as st

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from ngboost import NGBRegressor
import time




def run():
    # 데이터 로드
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    train_df = pd.read_csv('data/train_df.csv', encoding='utf-8-sig')
    test_df = pd.read_csv('data/test_df.csv', encoding='utf-8-sig')

    # 라벨 인코딩
    le = LabelEncoder()
    not_num_cols = train_df.select_dtypes(exclude='number').columns.to_list()

    for col in not_num_cols:
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.fit_transform(test_df[col])
    
    target = '가격(백만원)'
    X = train_df.drop(columns=[target])
    y = train_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 베이스라인 모델
    models = {
        "XGB": XGBRegressor(n_estimators=1000, random_state=42, objective='reg:squarederror'),  
        "CatBoost": CatBoostRegressor(n_estimators=1000, random_state=42, loss_function='RMSE', verbose=1),  
        "LGBM": LGBMRegressor(n_estimators=1000, random_state=42, objective='regression'),  
        "RandomForest": RandomForestRegressor(n_estimators=1000, max_depth=8, random_state=42, criterion='squared_error'),  
        "GradientBoosting": GradientBoostingRegressor(n_estimators=1000, random_state=42, criterion='squared_error'),  
        "ExtraTrees": ExtraTreesRegressor(n_estimators=1000, random_state=42),  
        "NGB": NGBRegressor(random_state=42) 
    }

    results = []

    for nm, model in models.items():
        print(f'{nm} 진행 중...')
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        elapsed_time = time.time() - start_time

        results.append({"model_nm": nm,
                        "RMSE": rmse,
                        "MAPE": mape,
                        "MAE": mae,
                        "R2": r2,
                        "duration": elapsed_time
                        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('models/model_results.csv', encoding='utf-8-sig', index=False)

    # 피클로 결과 저장
    with open('models/model_results.pkl', 'wb') as f:
        pickle.dump(results_df, f)

if __name__ == "__main__":
    run()