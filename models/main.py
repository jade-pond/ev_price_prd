import streamlit as st

import sys
import os
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
import optuna
import time



## 모델 페이지에 보여줄 것들을 불러옴 
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


    categorical_features = train_df.columns.to_list()

    categorical_features.remove('ID')
    categorical_features.remove('연식대비_보증기간_비율')
    categorical_features.remove('연식대비_주행거리_비율')
    categorical_features.remove('배터리용량대비_주행거리_비율')
    categorical_features.remove('가격(백만원)')
    categorical_features.remove('주행거리(km)')
    categorical_features.remove('배터리용량')


    def objective(trial, X_train, X_val, y_train, y_val, categorical_features):
        # 하이퍼파라미터 샘플링
        params = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
            "random_strength": trial.suggest_float("random_strength", 0, 1),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "verbose": 50,
            "random_seed": 42
        }

        # 모델 생성
        model = CatBoostRegressor(**params)
        
        # CatBoost Pool 사용
        train_pool = Pool(X_train, y_train, cat_features=categorical_features)
        valid_pool = Pool(X_val, y_val, cat_features=categorical_features)
        
        # 모델 학습
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)
        
        # 검증 데이터 예측 및 RMSE 계산
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        return rmse
    
    # Optuna 최적화 실행
    try:
        study = optuna.create_study(direction="minimize", 
                                    study_name="catboost_study",  
                                    storage="sqlite:///optuna_study.db" )
        study.optimize(lambda trial: objective(trial, 
                                               X_train, 
                                               X_val, 
                                               y_train, 
                                               y_val, 
                                               categorical_features), 
                                               n_trials=50)
    except: 
        print("already exist")
        # study = optuna.create_study(direction="minimize", 
        #                             study_name="catboost_study_2",  
        #                             storage="sqlite:///optuna_study_2.db" )
        # study.optimize(lambda trial: objective(trial,
        #                                      X_train, 
        #                                      X_val, 
        #                                      y_train, 
        #                                      y_val, 
        #                                      categorical_features), 
        #                                      n_trials=50)
    

    # 최적 하이퍼파라미터로 모델 학습
    best_model = CatBoostRegressor(**study.best_params)
    best_model.fit(
        Pool(X_train, y_train, cat_features=categorical_features),
        eval_set=Pool(X_val, y_val, cat_features=categorical_features),
        early_stopping_rounds=50
        )
    evals_result = best_model.get_evals_result()


    # 데이터 저장
    with open('data/split_data.pkl', 'wb') as f:
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

    # 모델 저장
    best_model.save_model("models/catboost_best_model.cbm")

    with open("models/catboost_best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open('models/catboost_loss.pkl', 'wb') as f:
        pickle.dump(evals_result, f)


if __name__ == '__main__':
    run()