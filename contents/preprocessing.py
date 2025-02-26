import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
# import koreanize_matplotlib
import pickle



@st.cache_data
def car_status(data):
    """
    Brand New: 0~ 10000 미만
    Nearly New: 10000~ 50000 미만
    Pre-Owned: 50000 ~ 
    """
    if data < 10000:
        return 'Brand New'
    elif 10000 <= data < 50000:
        return 'Nearly New'
    else:
        return 'Pre-Owned'
    


def feature_engineering (df):

    df['연식대비_보증기간_비율'] = df['보증기간(년)'] / (df['연식(년)'] + 1)
    df['연식대비_주행거리_비율'] = df['주행거리(km)'] / (df['연식(년)'] + 1)
    df['배터리용량대비_주행거리_비율'] = df['주행거리(km)'] / (df['배터리용량'] + 1)

    df = pd.get_dummies(df, columns=['제조사'], drop_first=True)
    df = pd.get_dummies(df, columns=['모델'], drop_first=True)
    df = pd.get_dummies(df, columns=['차량상태'], drop_first=True)
    df = pd.get_dummies(df, columns=['구동방식'], drop_first=True)
    df = pd.get_dummies(df, columns=['사고이력'], drop_first=True)
    df = pd.get_dummies(df, columns=['연식(년)'], drop_first=True)

    return df
    

def run():

    df = pd.read_csv('data/train.csv')
    df_1 = df.copy()

    # H사는 차량상태를 나누는 기준이 다른 기업과 상이함. 따라서 타사와 유사하게 기준을 맞춰줌.
    i = 0
    st.write(f'## ({i}) 차량상태 정규화')
    st.write("""
    ### 차량 상태와 주행 거리 분석
    - 차량 상태는 주행 거리와 밀접한 연관이 있는 것으로 확인.  
    - 특히, H사의 분류 기준은 타사와 다른 독특한 양상을 보임.
    
    *- Brand New*: 0 ~ 10,000 미만  
    *- Nearly New*: 10,000 ~ 50,000 미만  
    *- Pre-owned*: 50,000 이상 
            """)
    st.write(f'- 차량 상태를 주행 거리 기준으로 일관되게 조정(H사).')

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'*- Before: H사 pre-owned 기준은 11,477부터 시작함.*')
        car_status_df = df_1.groupby(['제조사','차량상태'])['주행거리(km)'].agg(['min','max'])
        st.dataframe(car_status_df)
        
    with col2:
        st.write(f'*- After: H사 pre-owned 기준을 50,000 이상으로 조정함.*')
        df_1['차량상태'] = df_1['주행거리(km)'].apply(car_status)
        car_status_df2 = df_1.groupby(['제조사','차량상태'])['주행거리(km)'].agg(['min','max'])
        st.dataframe(car_status_df2)


    
    # 결측치 처리 (배터리용량)
    ## 제조사, 모델, 차량 상태, 보증기간(년)에 따라 특정 배터리 용량 규격이 결정되는 경향
    ### 다만, 특정 모델의 경우 제조사, 모델, 차량 상태, 보증기간(년)이 동일하더라도 배터리 용량이 상이한 사례가 확인
    ### 이러한 경우, 특별한 규칙성이 발견되지 않아 최빈값으로 대체
    
    i += 1

    st.write(f'## ({i}) 배터리용량 결측치 처리') 
    st.write(f"""
        - 제조사, 모델, 차량 상태, 보증기간(년)이 동일한 모델의 배터리 용량을 참조하여 결측치를 보완.
        - 일반적으로, 제조사, 모델, 차량 상태, 보증기간에 따라 배터리 용량 규격이 결정되는 경향이 확인됨.
        - 그러나, 일부 모델에서는 동일한 조건(제조사, 모델, 차량 상태, 보증기간)을 가지더라도 배터리 용량이 상이한 사례가 존재.
        - 이러한 경우, 특별한 규칙성이 발견되지 않아 최빈값으로 대체.
            - 예: 동일한 조건에서 배터리 용량이 90과 93으로 나뉘는 경우, 최빈값(90)을 유지하고 나머지 값(93)은 제거.
        """)

    col1, col2 = st.columns(2)
    with col1:
        df_btr = df_1.dropna()
        df_btr = df_btr[['제조사','모델','차량상태','보증기간(년)','배터리용량']]
        st.code('''
                 df_btr = (df_btr.groupby(['제조사','모델','차량상태','보증기간(년)','배터리용량'])[['배터리용량']].agg(카운트=('배터리용량','count')))
                 ''', language="python")
        df_btr = df_btr.groupby(['제조사','모델','차량상태','보증기간(년)','배터리용량'])[['배터리용량']].agg(카운트=('배터리용량','count')).reset_index().sort_values(by=['제조사','모델','카운트'], ascending=False)
        df_btr.rename(columns={'배터리용량':'배터리'},inplace=True)
    
        st.write(f'- 중복 제거 전 df_btr.shape: {df_btr.shape}')
        st.dataframe(df_btr, hide_index=True)

    with col2:
        # 차량상태 데이터 정리
        # 가장 빈도가 높은 항목만 남기고 중복된 값을 제거
        st.code('''
                 df_btr = df_btr.drop_duplicates(subset=['제조사','모델','차량상태','보증기간(년)'])
                 ''', language="python")
        df_btr = df_btr.drop_duplicates(subset=['제조사','모델','차량상태','보증기간(년)'])
        # df_btr2.drop(columns='카운트', inplace=True)
        st.write(f'- 중복 제거 후 df_btr.shape: {df_btr.shape}')
        st.dataframe(df_btr, hide_index=True)

    st.write(f"""
        - 데이터 병합(Merge)
            - df에 df_btr를 병합. 
            - 제조사, 모델, 차량상태, 보증기간(년) 기준으로 left join 진행.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        df_1 = pd.merge(df_1, df_btr, on=['제조사','모델','차량상태','보증기간(년)'], how='left')
        st.write(f'- 병합 후')
        st.code('''
                df_1 = pd.merge(df_1, df_btr, on=['제조사','모델','차량상태','보증기간(년)'], how='left')
                 ''',language="python")
        st.write(f'- df_1.head()')
        st.dataframe(df_1, hide_index=True)
        st.dataframe(df_1.isna().sum())
    with col2:
        st.write(f"- \'배터리용량\'결측치 \'배터리\' 값 대체")
        df_1.loc[df_1['배터리용량'].isna(),'배터리용량'] = df_1.loc[df_1['배터리용량'].isna(),'배터리']
        df_1.drop('배터리',axis=1, inplace=True)
        st.code('''
            df_1.loc[df_1['배터리용량'].isna(),'배터리용량'] = df_1.loc[df_1['배터리용량'].isna(),'배터리']
            ''',language="python")
        st.write(f'- df_1.head()')
        st.dataframe(df_1, hide_index=True)
        st.dataframe(df_1.isna().sum())

    st.write(f"""
             - 병합 후에도 처리되지 않은 15개의 결측치 확인. 
             - 해당 데이터는 다음과 같은 공통점을 가짐.
                - 제조사: T사 
                - 모델: MX
                - 차량상태: Nearly New
                - 보증기간: 4년
             """)
    st.dataframe(df_1[df_1['배터리용량'].isna()], hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        mx_df = df_1[(df_1['제조사'] =='T사')&
        (df_1['모델'] =='MX')&
        (df_1['차량상태']=='Nearly New')
        ][['모델','배터리용량','보증기간(년)']].drop_duplicates().sort_values(by='보증기간(년)')
        st.write('T사 MX 모델 Nearly New의 배터리용량')    
        st.dataframe(mx_df, hide_index=True)
    with col2:
        mx_df['배터리용량'].interpolate(method='linear', inplace=True)
        st.write('interpolate linear 방식으로 결측치 처리')
        st.dataframe(mx_df, hide_index=True)

    df_1.loc[(df_1['제조사'] =='T사')&
     (df_1['모델'] =='MX')&
     (df_1['차량상태']=='Nearly New')&
     (df_1['배터리용량'].isna()),'배터리용량'] = mx_df.loc[mx_df['보증기간(년)'] == 4,'배터리용량'].iloc[0]
    
    # df_1.to_csv('data/df_train_preprocessed.csv', encoding='utf-8-sig', index=False)

    i += 1
    st.write(f'## ({i}) 피처 엔지니어링')
    st.write(f"""
             
    1. 피처 생성
    - 비율 변수 생성:
        - 연식대비_보증기간_비율: 보증기간/연식
        - 연식대비_주행거리_비율: 주행거리/연식
        - 배터리용량대비_주행거리_비율: 주행거리/배터리용량
    
    2. 카테고리 변수 더미 데이터 생성
    - 카테고리 변수 더미 변수 생성: 제조사, 모델, 차량상태, 구동방식, 사고이력, 연식(년).

    """)
    df_1.drop(columns=['카운트'], inplace=True)

    train_df = df_1.copy()
    train_df = feature_engineering(train_df)

    test_df = pd.read_csv('data/test.csv', encoding='utf-8-sig')
    test_df = feature_engineering(test_df)

    train_df.to_csv('data/train_df.csv', encoding='utf-8-sig', index=False)
    test_df.to_csv('data/test_df.csv', encoding='utf-8-sig', index=False)

