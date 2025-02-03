project/
│
├── app.py                      # 메인 Streamlit 파일
├── contents/                   
│   ├── data_description.py     # 데이터 설명
│   ├── eda.py                  # EDA
│   ├── preprocessing.py        # 전처리, 피처엔지니어링
│   ├── model.py                # 모델 예측
│
├── data/                     
│   ├── train.csv
│   ├── test.csv
│
├── models/                    
│   ├── __init__.py         
│   ├── baseline.py             # xgb, lgbm, catboost, randomfoest, gradient boosting, extratrees, ngb 베이스라인 예측 모듈
│   ├── main.py                 # catboost 예측 모델 및 optuna 하이퍼파라미터 튜닝
│
└── utils/                   
    ├── btr_feature_importances.py   # 제조사별 배터리용량 예측
    ├── btr_xgb_model.pkl
    └── btr_feature_importances.py
