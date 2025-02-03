import streamlit as st


def run():
    st.write("## 분석 요약")
    st.write(f"""
    
    ### 1. 데이터 전처리 (Data Preprocessing)
    - **결측치 처리**: 배터리 용량
        - 제조사, 모델, 차량 상태, 보증기간(년)이 동일한 데이터의 평균값으로 대체.
        - 위 방법으로 해결되지 않은 15개 샘플(T사 MX 모델, Nearly New, 보증기간 4년)에 대해서는, 동일 모델의 보증기간 1~3년 데이터를 기준으로 보간법 (interpolation) 적용하여 보완.
    - **데이터 타입 변환**
        - 범주형 변수는 **One-hot encoding** (`pandas.get_dummies`) 및 **Label Encoding** (`sklearn.preprocessing.LabelEncoder`) 적용.
        - 보증기간 및 연식 변수는 `nunique < 5` 조건을 만족하여 범주형 변수로 변환하여 처리.
    
    ### 2. 모델링 (Modeling)
    - **베이스라인 모델**
        - XGBoost (`XGBRegressor`)
        - LightGBM (`LGBMRegressor`)
        - CatBoost (`CatBoostRegressor`)
        - RandomForest
        - GradientBoosting
        - ExtraTrees
        - NGBoost (`NGBRegressor`)
    - **최종 선정 모델: CatBoostRegressor**
        - 탐색적 데이터 분석(EDA)에서 카이제곱 검정 결과, **가격(Target) 변수는 제조사, 모델과 가장 높은 상관성을 보임**.
        - 주요 변수(제조사, 모델, 차량 상태, 구동 방식, 사고 이력, 보증기간, 연식) 대부분이 **범주형 변수**로 구성됨.
        - 범주형 변수에 강한 성능을 보이는 **CatBoostRegressor**가 가장 적합한 모델로 판단.

    ### 3. 하이퍼파라미터 튜닝 (Hyperparameter Optimization)
    - **Optuna 기반 자동 하이퍼파라미터 탐색 적용**
    - **파라미터 영향도 분석**: Feature Importance 분석 결과, **learning rate가 성능에 가장 큰 영향을 미치는 주요 하이퍼파라미터로 확인됨**. 이를 중심으로 최적화 진행.

    ### 4. 최종 성능 평가 (Performance Evaluation)
    - **Root Mean Squared Error (RMSE):** `1.2849`  
    - **Private Leaderboard Score:** `54`  
    - **결론:** 최적화된 CatBoost 모델을 통해 예측 성능을 향상시켰으며, **주요 변수를 범주형 처리하는 것이 모델 성능에 긍정적인 영향을 미쳤음을 확인**.
    
    """)
