"""소프트웨어 결함 예측 프로그램의 홈 화면 및 페이지 라우터."""
import streamlit as st

from common import DATA_URL, DATA_SOURCE, TARGET, load_record

st.set_page_config(
    page_title="소프트웨어 결함 예측 프로그램",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)


def home_page():
    """사이트 설명과 이용 방법을 안내하는 시작 화면."""
    st.title("🧩 소프트웨어 결함 예측 프로그램")
    st.caption("NASA PROMISE Software Defect Dataset · JM1 기반")

    st.markdown(
        "이 프로그램은 소프트웨어 모듈의 코드 특성 데이터를 머신러닝 모델로 분석하여 "
        "**해당 모듈에 결함이 있을 가능성**을 예측하는 프로젝트입니다. "
        "데이터 분석부터 모델 학습, 성능 평가, 사용자 입력 예측까지 단계별로 이용할 수 있습니다."
    )

    st.divider()
    st.subheader("프로그램에서 할 수 있는 일")
    cards = st.columns(2)
    with cards[0]:
        st.markdown("#### 🧹 데이터 분석 및 전처리")
        st.write("JM1 데이터의 열 이름과 자료형, 결측값, 중복 데이터, 결함 분포를 확인합니다.")
        st.markdown("#### 🧠 머신러닝 모델 제작")
        st.write("로지스틱 회귀 또는 랜덤 포레스트를 선택해 데이터를 8:2로 나누어 학습합니다.")
    with cards[1]:
        st.markdown("#### 📊 모델 성능 평가")
        st.write("Accuracy, Precision, Recall, F1-score와 혼동 행렬을 확인하고 임계값별 결과를 비교합니다.")
        st.markdown("#### 🔍 사용자 코드 결함 예측")
        st.write("코드 특성값을 입력해 결함 예측 확률과 임계값에 따른 분류 결과를 확인합니다.")

    st.divider()
    st.subheader("이용 방법")
    st.markdown(
        "1. 사이드바의 **모델 개발 → 데이터 분석 및 전처리**에서 데이터 구조와 결함 분포를 확인합니다.\n"
        "2. **머신러닝 모델 제작**에서 모델을 선택하고 학습을 시작합니다. 학습이 완료되면 임계값을 조절할 수 있습니다.\n"
        "3. **모델 성능 평가**에서 임계값별 지표와 혼동 행렬을 비교합니다.\n"
        "4. **사용자 기능 → 사용자 코드 결함 예측**에서 코드 특성값을 입력해 결과를 확인합니다."
    )
    st.info(
        "페이지 사이를 이동해도 학습된 모델과 임계값은 현재 Streamlit 세션 상태에 유지됩니다. "
        "브라우저를 새로고침하거나 세션이 종료되면 모델은 초기화될 수 있으므로 다시 학습해야 합니다."
    )

    st.divider()
    st.subheader("예측 기준 알아보기")
    st.markdown(
        "**예측 대상:** `defects`  \n"
        "- `True` 또는 `1`: 결함 있음  \n"
        "- `False` 또는 `0`: 결함 없음"
    )
    st.markdown(
        "**임계값(Threshold)**은 모델이 계산한 결함 발생 확률을 결함 있음/없음으로 구분하는 기준입니다. "
        "예를 들어 임계값이 0.30이면 결함 발생 확률이 30% 이상일 때 결함 있음으로 분류합니다. "
        "임계값을 낮추면 결함을 더 많이 찾아내는 경향이 있지만, 정상 모듈을 결함으로 잘못 판단하는 경우도 늘 수 있습니다."
    )

    st.divider()
    st.subheader("데이터 출처와 기록")
    st.write(f"**데이터:** {DATA_SOURCE}")
    st.markdown(f"**CSV 주소:** [{DATA_URL}]({DATA_URL})")
    st.write(f"**예측 대상 열:** `{TARGET}`")
    st.write(
        "변수의 우리말 뜻과 데이터 출처는 `내_기록/변수_뜻과_데이터_출처.json`에 저장됩니다. "
        "기록 파일과 폴더는 처음 실행할 때 자동으로 생성되며, 상세 내용은 데이터 분석 및 전처리 페이지에서 편집할 수 있습니다."
    )
    st.warning(
        "이 프로그램은 코드의 통계적 특성을 바탕으로 결함 가능성을 예측합니다. "
        "실제 버그가 있는 코드 줄이나 위치를 찾아내지는 않으며, 예측 결과만으로 결함 유무를 확정해서는 안 됩니다."
    )

    # 처음 방문했을 때 JSON 기록 폴더/파일을 생성하고 세션에 불러온다.
    if "record" not in st.session_state:
        st.session_state["record"] = load_record()

    st.divider()
    st.subheader("바로 이동하기")
    quick_links = st.columns(2)
    with quick_links[0]:
        st.page_link(processing_page, label="데이터 분석 및 전처리", icon="🧹", use_container_width=True)
        st.page_link(make_page, label="머신러닝 모델 제작", icon="🧠", use_container_width=True)
    with quick_links[1]:
        st.page_link(rating_page, label="모델 성능 평가", icon="📊", use_container_width=True)
        st.page_link(predict_page, label="사용자 코드 결함 예측", icon="🔍", use_container_width=True)


# st.Page와 st.navigation으로 하위 폴더의 페이지를 직접 등록한다.
home = st.Page(home_page, title="프로그램 소개", icon="🏠", default=True)
processing_page = st.Page("pages/model/processing.py", title="데이터 분석 및 전처리", icon="🧹")
make_page = st.Page("pages/model/make.py", title="머신러닝 모델 제작", icon="🧠")
rating_page = st.Page("pages/model/rating.py", title="모델 성능 평가", icon="📊")
predict_page = st.Page("pages/use/predict.py", title="사용자 코드 결함 예측", icon="🔍")

navigation = st.navigation(
    {
        "홈": [home],
        "모델 개발": [processing_page, make_page, rating_page],
        "사용자 기능": [predict_page],
    },
    position="sidebar",
)
navigation.run()
