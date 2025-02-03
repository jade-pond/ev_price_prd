import streamlit as st

import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import optuna
from optuna.visualization.matplotlib import plot_param_importances


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

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.baseline import run as baseline_run
from models.main import run as main_run

def pre_run():
    if os.path.exists('models/catboost_loss.pkl'):
        pass
    else:
        baseline_run()
        main_run()

## 모델 페이지에 보여줄 것들을 불러옴 
def run():
    # 베이스라인의 결과를 보여줌
    baseline_df = pd.read_csv('models/model_results.csv', encoding='utf-8-sig')
    st.subheader('모델별 베이스라인 비교')
    st.write('- Catboost Regressor가 전반적으로 우수한 성능을 보임')
    st.dataframe(baseline_df)

    st.subheader('모델별 성능 비교: RMSE')
    # st.write('- Catboost Regressor가 전반적으로 우수한 성능을 보임')
    plt.figure(figsize=(8,6))
    sns.barplot(data=baseline_df, 
            x='model_nm', 
            y='RMSE', 
            hue='model_nm', 
            # palette='PiYG',  
            palette = 'cubehelix')
    ax = plt.gca()  
    ax.set_facecolor("white") 
    st.pyplot(plt)
    
    st.write('## 최종 모델: Catboost Regressor')
    # st.write('### work flow')
    # st.write('데이터 수집 - 데이터 전처리 - 모델 선정(catboost) - 하이퍼파라미터튜닝(optuna)')
    image_url = "https://www.researchgate.net/publication/372827215/figure/fig3/AS:11431281218024088@1705462528705/Structure-of-CatBoost-algorithm.png"
    st.image(image_url, caption="Catboost 알고리즘 ", use_container_width=True)


    # with open('models/model_results.pkl', 'rb') as f:
    #     result_df = pickle.load(f)

    st.subheader('손실함수: Training vs Validation')
    st.write("""
    - 학습 과정에서의 훈련 데이터와 검증 데이터의 손실 함수의 변화를 나타냄.
    - 초반에는 손실값이 급격히 감소하며 모델이 데이터를 학습하는 모습을 보여줌.
    - 반복이 진행됨에 따라 손실값은 안정화됨. 
    - 일반화 성능: 훈련 데이터와 검증 데이터 간의 차이가 거의 없음.
    """)


    with open('models/catboost_loss.pkl', 'rb') as f:
        evals_result = pickle.load(f)

    # 그래프 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(evals_result['validation']['RMSE'], label='Validation Loss (RMSE)', color='blue')
    plt.plot(evals_result['learn']['RMSE'], label='Training Loss (RMSE)', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    # plt.title('손실함수: Training vs Validation')
    plt.legend()
    ax = plt.gca()  # 현재 Axes 객체 가져오기
    ax.set_facecolor("white")  # 배경색을 흰색으로 설정
    st.pyplot(plt)
    plt.close()  # Streamlit에서 그래프가 중복되지 않도록 추가




    # 피처 임포턴스
    with open('models/catboost_best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)

    feature_importances = best_model.get_feature_importance()
    feature_names = best_model.feature_names_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
    
    feature_importance_df.to_csv('models/feature_importance.csv', index=False, encoding='utf-8-sig')
    
    st.subheader('피처 중요도')
    st.write("""
        - 제조사별 평균 가격 정보를 반영한 '제조사_그룹' 변수가 가장 높은 중요도를 나타냄. 
        - 특히, 상대적으로 높은 가격대를 보였던 제조사 'P사' 관련 변수 또한 높은 영향력을 보였음.
        """)
    
    feature_importance_df = pd.read_csv('models/feature_importance.csv')
    
    fig = px.bar(feature_importance_df,
                 x='Importance', 
                 y='Feature')
    
    st.plotly_chart(fig)




    

    
    # Optuna 스터디 불러오기
    study_name = "catboost_study"  # 스터디 이름
    storage_name = "sqlite:///optuna_study.db"  # SQLite 파일 경로
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    # **1. Hyperparameter Importances**
    st.subheader("하이퍼파라미터 중요도")
    plt.figure(figsize=(8,6))
    fig_importances = optuna.visualization.matplotlib.plot_param_importances(study)
    fig_importances.patch.set_facecolor("white")
    st.pyplot(plt)

    # # **2. Optimization History**
    st.subheader("최적화 히스토리")
    plt.figure(figsize=(8,6))
    fig_optimization = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig_optimization.patch.set_facecolor("white")
    st.pyplot(plt)


    # 최종 모델의 RMSE 및 최적 하이퍼파라미터
    st.subheader("최종 모델 요약")

    # 최적화 결과 요약
    best_params = study.best_params  # 최적의 하이퍼파라미터
    best_rmse = study.best_value  # 최적의 RMSE 값

    # 하이퍼파라미터와 RMSE를 DataFrame으로 정리
    summary_df = pd.DataFrame({
        "Parameter": list(best_params.keys()) + ["Best RMSE"],
        "Value": list(best_params.values()) + [best_rmse]
    })

    # Streamlit 표 출력
    st.write("최적화된 하이퍼파라미터와 최종 RMSE:")
    st.dataframe(summary_df)  

    
