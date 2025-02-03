import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
# import koreanize_matplotlib
import pickle
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


warnings.filterwarnings('ignore')

# #####

# font_path = "NanumGothic.ttf"  # 폰트 경로
# font_prop = fm.FontProperties(fname=font_path)
# plt.rc("font", family=font_prop.get_name())
# plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

# #####

# 데이터 로드 함수
@st.cache_data
def load_data(filepath):
    return pd.read_csv(filepath)

# 카이제곱 검정 및 히트맵 생성 함수
def create_chi2_heatmap_with_plotly(df, categoric_vars):

    df = df.dropna(axis=0)
    corr_matrix = pd.DataFrame(index=categoric_vars, columns=categoric_vars)
    for i, var1 in enumerate(categoric_vars):
        for var2 in categoric_vars[i+1:]:
            cross_tab = pd.crosstab(df[var1], df[var2])
            chi2, p, dof, _ = chi2_contingency(cross_tab)
            n = cross_tab.sum().sum()
            min_dim = min(cross_tab.shape) - 1
            corr_matrix.loc[var1, var2] = np.sqrt(chi2 / (n * min_dim))
            corr_matrix.loc[var2, var1] = np.sqrt(chi2 / (n * min_dim))
    corr_matrix.fillna(0, inplace=True)

    # 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.to_list(),
        y=corr_matrix.index.to_list(),
        # colorscale='coolwarm',
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        colorbar=dict(title="상관계수")
    ))

    fig.update_layout(
        title="카이제곱 히트맵",
        xaxis=dict(title="변수"),
        yaxis=dict(title="변수"),
        font=dict(family="NanumGothic, sans-serif")  # 한글 폰트 설정
    )

    # Streamlit에 출력
    st.plotly_chart(fig)



def making_bins(df, col, bins=10):
    df[f'binned_{col}'] = pd.qcut(df[col], q=bins, labels=[i for i in range(1,bins+1)], duplicates='drop')
    return df

# EDA 실행 함수
def run():    
    st.title("EDA")
    df = load_data("data/train.csv")
    df_2 = df.copy()
    tab1, tab2 = st.tabs(["전체 EDA", "제조사별 EDA"])

    with tab1:
        # point1: 카이제곱 검정 히트맵
        i = 0
        i+=1
        st.write(f"### ({i}) 카이제곱 검정 기반 히트맵")
        st.write("- \'배터리용량\'은 \'차량상태, 모델\'과 가장 상관성이 큰 것으로 보여짐")

        ## 히트맵 생성을 위한 수치형변수 범주화
        numeric_cols = ['배터리용량','주행거리(km)','가격(백만원)']
        for col in numeric_cols:
            making_bins(df_2, col, bins=4)
        categoric_vars = ['제조사', '모델', '차량상태','구동방식', '보증기간(년)', '사고이력', '연식(년)', 'binned_배터리용량', 'binned_주행거리(km)','binned_가격(백만원)']
        create_chi2_heatmap_with_plotly(df_2, categoric_vars)

        
        # point2: 차량상태 - 주행거리
        i+=1
        st.write(f"### ({i}) \'주행거리\'와 \'차량상태\' 간 관계")
        st.write("- 차량상태는 주행거리에 의해 결정되는 경향을 보이는 것으로 보여짐.")
        fig = px.scatter(df, 
                         x="주행거리(km)", 
                         y="차량상태", 
                         color="차량상태",  
                         color_discrete_sequence=px.colors.qualitative.Set2, )
        fig.add_vline(x=10000, line_dash="dash", line_color="grey", line_width=2)
        fig.add_vline(x=50000, line_dash="dash", line_color="grey", line_width=2)
        fig.update_layout(plot_bgcolor="white", xaxis=dict(title="주행거리(km)", showgrid=True), yaxis=dict(title="차량상태", showgrid=True))
        st.plotly_chart(fig)

        # point3: 차량상태 - 배터리용량
        i+=1
        st.write(f"### ({i}) \'배터리용량\'과 \'주행거리\' 및  \'차량상태\' 간 관계")
        st.write("- 차량상태가 Brand New일 경우 배터리용량은 90 이상임. 즉, 신형일수록 배터리용량이 높음.")
        status_usage = (df.groupby('차량상태')['주행거리(km)']
                        .agg(['min', 'max'])
                        .rename(columns={'min': '주행거리 min', 'max': '주행거리 max'})
                        .reset_index())
        fig = px.scatter(df, 
                         x='주행거리(km)',     
                         y='배터리용량', 
                         color='차량상태',
                         color_discrete_sequence=px.colors.qualitative.Set2, )
        fig.add_hline(y=90, line_color='grey', line_dash='dash', line_width=2)
        st.plotly_chart(fig)

        
        # point3: 배터리용량 - 주행거리(km)
        i += 1
        st.write(f"### ({i}) \'배터리용량\'과 \'주행거리(km)\' 간 관계")
        st.write("- \'주행거리(km)\': \'주행거리(km)\'가 0에 가까운 경우(새 차량 또는 사용 초기), \'배터리용량\'이 대부분 90~100 사이에 위치.")
        st.write("- \'주행거리(km)\': \'주행거리(km)\'가 늘어나면서 \'배터리 용량\'이 전반적으로 감소하는 경향을 보임.")
        fig = px.line(df, 
                      x='주행거리(km)', 
                      y='배터리용량', 
                      color='차량상태', 
                      color_discrete_sequence=px.colors.qualitative.Set2,)
        fig.add_hline(y=90, line_color='red', line_dash='dash', line_width=2)
        st.plotly_chart(fig)

        
        ## point4: 배터리용량 - 보증기간
        i+=1
        st.write(f"### ({i}) \'배터리용량\'과 \'보증기간(년)\' 간 관계")
        st.write("- \'보증기간(년)\': \'보증기간(년)\'이 9 혹은 10일 경우 배터리용량은 90. ")
        fig = px.box(df, 
                     x='보증기간(년)', 
                     y='배터리용량',
                     color_discrete_sequence=px.colors.qualitative.Set2, )
        fig.add_hline(y=90, line_color='red', line_dash='dash', line_width=2)
        st.plotly_chart(fig)

        
        ## point5: 가격 - 보증기간(년)
        i+=1
        price_mean = df['가격(백만원)'].mean()
        st.write(f"### ({i}) \'가격(백만원)\'과 \'보증기간(년)\' 간 관계")
        st.write("- \'보증기간\'이 2년인 경우 가격 레인지가 높음.")
        fig = px.box(df, 
                     x='보증기간(년)', 
                     y='가격(백만원)',
                     color_discrete_sequence=px.colors.qualitative.Set2,)
        fig.add_hline(y=price_mean, line_color = 'black', line_dash='dash', line_width=2)
        st.plotly_chart(fig)


        ## point6: 가격 - 제조사
        i+=1
        st.write(f"### ({i}) \'가격(백만원)\'과 \'제조사\' 간 관계")
        st.write("- \'가격(백만원)\': \'가격(백만원)\'의 range는 \'제조사\'별로 상당한 차이가 있음.")
        fig = px.box(df, 
                     x='제조사', 
                     y='가격(백만원)', 
                     color='제조사',
                     color_discrete_sequence=px.colors.qualitative.Set2,)
        fig.add_hline(y=price_mean, line_color = 'black', line_dash='dash', line_width=2)
        st.plotly_chart(fig)

        ## point6: 가격 - 모델
        i += 1
        st.write(f"### ({i}) \'가격(백만원)\'과 \'모델\' 간 관계")
        st.write("- \'가격(백만원)\': \'가격(백만원)\'의 range는 \'모델\'별로 상당한 차이가 있음.")
        fig = px.box(df, 
                     x='모델', 
                     y='가격(백만원)', 
                     color='제조사',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.add_hline(y=price_mean, line_color = 'black', line_dash='dash', line_width=2)
        st.plotly_chart(fig)

        ## point7: 가격 - 구동방식
        i += 1
        st.write(f"### ({i}) \'가격(백만원)\'과 \'구동방식\' 간 관계")
        st.write("- \'가격(백만원)\': \'구동방식\'이 FWD일 경우 가격이 비교적 저렴한 구간에 속함.")
        fig = px.box(df, 
                     x='구동방식', 
                     y='가격(백만원)',
                     color='구동방식',
                     color_discrete_sequence=px.colors.qualitative.Set2,)
        fig.add_hline(y=price_mean, line_color = 'black', line_dash='dash', line_width=2)
        st.plotly_chart(fig)



    
    # 개별 현황 탭
    with tab2:
        st.subheader("제조사별 가격 데이터 시각화")
        company_ls = ['P사' ,'K사', 'A사', 'B사' ,'H사', 'T사', 'V사']
        company_dict = {firm : df[df['제조사'] == firm] for firm in company_ls}
        selected_company = st.selectbox("제조사를 선택하세요", options=company_ls)
        selected_df = company_dict[selected_company]

        # 결측치 현황
        st.subheader("결측치 현황")
        st.write(f"{selected_company} df shape: {selected_df.shape}")
        st.write(f"{selected_company} 결측치: {selected_df['배터리용량'].isna().sum()}")
        isna_df = selected_df[selected_df['배터리용량'].isna()][['차량상태', '모델']].value_counts().reset_index()
        isna_df.columns = ['차량상태', '모델', '결측치 개수']
        st.dataframe(isna_df, hide_index=True)

        # 차량상태 및 모델별 배터리용량 분포
        st.header(f'{selected_company} 차량상태 및 모델 별 배터리용량 및 개수')
        df_summary = selected_df.groupby(['차량상태', '모델', '배터리용량'])['배터리용량'].count().reset_index(name='개수')
        st.dataframe(df_summary, hide_index=True)

        # 전체적인 배터리용량 분포
        st.subheader(f'{selected_company} 모델별 배터리용량 분포')
        fig = make_subplots(rows=1, cols=2)
        fig = px.scatter(selected_df, 
                         x='모델', 
                         y='배터리용량', 
                         color='모델')
        fig = px.box(selected_df, 
                     x='모델', 
                     y='배터리용량', 
                     color='모델')
          
        st.plotly_chart(fig)
        plt.close()

        # 배터리용량 피처 중요도
        st.subheader(f'{selected_company}의 배터리용량 피처 중요도(model: XGBRegressor)')
        with open('utils/btr_feature_importances.pkl', 'rb') as fi_file:
            feature_importances = pickle.load(fi_file)

        col3, col4 = st.columns(2)
        with col3:
            st.dataframe(feature_importances.head(5), hide_index=True)
        with col4:
            fig = px.bar(feature_importances,
                         x='Feature', 
                         y='Importance')
            st.plotly_chart(fig)
            
