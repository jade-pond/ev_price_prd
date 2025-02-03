import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
# import koreanize_matplotlib
import pandas as pd
import numpy as np
import os

import matplotlib.font_manager as fm
import warnings

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 데이터 로드 함수
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)


def run():
    st.title("데이터 설명")
    df = load_data("data/train.csv")

    # 0.데이터 정보
    i = 0
    st.write(f"""
             ### ({i}) 데이터셋 설명
             - 데이터 출처: Dacon
             - url: https://dacon.io/competitions/official/236424/data
             - 데이터 설명
                - 제조사: EV 제조사
                - 모델: EV 모델
                - 차량상태: 새것, 거의 새것, 중고
                - 구동방식: 차량 구동 방식 
                - 주행거리(km): 차량이 주행한 거리 
                - 보증기간(년): 차량의 보증기간 
                - 사고이력: 사고 유무 (Yes/No) 
                - 연식(년): 차량의 연식 1~7
                - 가격(백만원): 차량의 가격.             
            
             """)

    i += 1
    st.write(f"""
             ### ({i}) 데이터 구조
             - 총 결측치 수: {df.isna().sum().sum()}
             - df shape: {df.shape}
             """)
    col1, col2 = st.columns(2)
    with col1:
        info_df = pd.DataFrame({
            '칼럼명': df.columns,
            'Non-Null 개수': df.notnull().sum(),
            '데이터 타입': df.dtypes
        })
        info_df = info_df.reset_index()
        info_df = info_df.iloc[:,1:]
        st.dataframe(info_df, hide_index=False)

    with col2:
        missing_values = df.isna().astype('int').T
        fig = go.Figure(data=go.Heatmap(
            z=missing_values.values, 
            x=missing_values.columns,
            y=missing_values.index,  
            # colorscale="viridis",  # 색상 팔레트
            showscale=False  ))
        
        st.plotly_chart(fig)

