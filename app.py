import streamlit as st
from streamlit_option_menu import option_menu

# 페이지 임포트
from contents.eda import run as eda_run
from contents.summary import run as summary_run
from contents.preprocessing import run as preprocessing_run
from contents.model import pre_run as model_prerun
from contents.model import run as model_run

from contents.data_description import run as data_description_run


MENU_DICT = {
    "요약": summary_run,
    "데이터 설명": data_description_run,
    "EDA": eda_run,
    "전처리": preprocessing_run,
    "예측 모델": [model_prerun,model_run,]
    # ...
}

def main():
    with st.sidebar:
        selected = option_menu(
            "메뉴",
            ["요약",
             "데이터 설명", 
             "EDA", 
             "전처리", 
             "예측 모델"],  # ...
            icons=[ "card-text",
                   "database" ,
                   "bar-chart",
                   "funnel",
                   "robot"],
            menu_icon="cast",
            default_index=0
        )
    # 선택된 페이지 함수 실행
    if selected == "예측 모델":
        for func in MENU_DICT[selected]:  # 리스트 내 모든 함수 실행
            func()
    else:
        MENU_DICT[selected]()  # 일반적인 경우 실행

if __name__ == "__main__":
    main()
