import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    DATA_URL, DATA_SOURCE, TARGET, DEFAULT_MEANINGS,
    load_app_data, load_record, save_record, show_record_editor,
    train_and_evaluate, evaluate_at_threshold, get_feature_types,
    drop_high_correlation_features
)

st.set_page_config(
    page_title="소프트웨어 결함 예측 프로그램",
    page_icon="🧩",
    layout="wide",
)


def page_data_analysis(raw_df, cleaned_df, feature_columns, duplicate_count,
                       invalid_target_count, record):
    st.header("1. 데이터 분석 및 전처리")
    st.write(
        "NASA PROMISE JM1 데이터의 열 구성, 결측값, 중복 행과 결함 분포를 확인합니다. "
        "실제 CSV의 열 이름과 자료형을 그대로 사용합니다."
    )

    row1 = st.columns(4)
    row1[0].metric("원본 데이터 행", f"{len(raw_df):,}개")
    row1[1].metric("원본 데이터 열", f"{len(raw_df.columns):,}개")
    row1[2].metric("완전히 동일한 중복 행", f"{duplicate_count:,}개")
    row1[3].metric("제거한 결측/비정상 예측 대상", f"{invalid_target_count:,}개")

    st.subheader("데이터 열과 자료형 확인")
    type_table = pd.DataFrame(
        {
            "열 이름": raw_df.columns,
            "실제 자료형": [str(raw_df[column].dtype) for column in raw_df.columns],
            "결측값 개수": [int(raw_df[column].isna().sum()) for column in raw_df.columns],
            "역할": ["예측 대상(Target)" if column == TARGET else "입력 변수(Feature)"
                    for column in raw_df.columns],
        }
    )
    st.dataframe(type_table, use_container_width=True, hide_index=True)

    with st.expander("원본 데이터 미리 보기"):
        st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("입력 변수와 예측 대상 구분")
    left, right = st.columns([3, 1])
    with left:
        st.markdown("**입력 변수(독립 변수, X)**")
        st.write(", ".join(feature_columns))
    with right:
        st.markdown("**예측 대상(종속 변수, y)**")
        st.code(TARGET)
        st.caption("True = 결함 있음 / False = 결함 없음")

    st.subheader("결측값과 중복 데이터")
    before_missing = int(raw_df.isna().sum().sum())
    after_missing_features = int(cleaned_df[feature_columns].isna().sum().sum())
    before_after = pd.DataFrame(
        {
            "항목": ["행 개수", "열 개수", "전체 결측값 개수", "완전 중복 행"],
            "전처리 전": [len(raw_df), len(raw_df.columns), before_missing, duplicate_count],
            "전처리 후": [
                len(cleaned_df),
                len(cleaned_df.columns),
                after_missing_features,
                0,
            ],
        }
    )
    st.dataframe(before_after, use_container_width=True, hide_index=True)
    st.caption(
        "전처리 후 결측값 개수는 입력 변수에 남아 있는 결측값입니다. "
        "입력 변수 결측값은 데이터 전체의 통계로 미리 채우지 않고, "
        "모델 학습 시 학습 데이터의 중앙값(숫자형) 또는 최빈값(범주형)으로 처리합니다."
    )
    if invalid_target_count:
        st.info(
            f"예측 대상 '{TARGET}'이 비어 있거나 True/False로 해석되지 않는 "
            f"{invalid_target_count}개 행은 학습에 사용할 수 없어 제외했습니다."
        )

    st.subheader("전처리 방법")
    st.markdown(
        "- **중복 데이터:** 모든 열의 값이 완전히 동일한 행을 제거합니다.\n"
        "- **예측 대상 결측값:** `defects`가 비었거나 True/False로 바꿀 수 없는 행은 제외합니다.\n"
        "- **숫자형 입력 변수:** 학습 데이터의 중앙값으로 결측값을 채웁니다.\n"
        "- **문자열/범주형 입력 변수:** 학습 데이터의 최빈값으로 결측값을 채우고 원-핫 인코딩합니다.\n"
        "- **데이터 누수 방지:** `defects`는 X에서 제외하며, 결측값 처리·인코딩·표준화는 학습 데이터에만 맞춥니다."
    )

    st.subheader("결함 있음/없음 데이터 분포")
    target_counts = cleaned_df[TARGET].value_counts().reindex([0, 1], fill_value=0)
    chart_df = pd.DataFrame(
        {
            "분류": ["결함 없음 (False)", "결함 있음 (True)"],
            "데이터 개수": [int(target_counts.loc[0]), int(target_counts.loc[1])],
        }
    )
    fig = px.bar(
        chart_df,
        x="분류",
        y="데이터 개수",
        text="데이터 개수",
        title="JM1 데이터의 결함 분포",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="데이터 개수", xaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    total = int(target_counts.sum())
    if total:
        defect_rate = target_counts.loc[1] / total * 100
        st.info(
            f"결함 있음 비율은 {defect_rate:.1f}%입니다. 두 분류의 개수 차이가 크면 "
            "정확도만으로 모델을 판단하지 않는 것이 중요합니다."
        )
        
    st.divider()
    st.subheader("📊 데이터 상관관계 및 다중공선성(중복 변수) 제거")
    st.write(
        "변수들 간의 상관관계를 히트맵으로 확인하고, 상관계수가 너무 높은 "
        "(다중공선성이 있는) 중복 변수를 모델 학습에서 제외합니다."
    )

    # 임계값 슬라이더 (session_state에 저장하여 모델 학습 페이지와 연동)
    if "corr_threshold" not in st.session_state:
        st.session_state["corr_threshold"] = 0.90
    
    threshold = st.slider(
        "상관계수 임계값 (이 값 이상이면 중복으로 간주하고 제거)", 
        min_value=0.50, max_value=1.00, value=st.session_state["corr_threshold"], step=0.05
    )
    st.session_state["corr_threshold"] = threshold

    reduced_features, dropped_features, _ = drop_high_correlation_features(
        cleaned_df, feature_columns, threshold=threshold
    )

    # 히트맵 시각화 (원본 부호가 있는 상관계수를 보여주기 위해 절댓값 적용 전 계산)
    numeric_columns, _ = get_feature_types(cleaned_df, feature_columns)
    raw_corr = cleaned_df[numeric_columns].corr()

    fig_corr = px.imshow(
        raw_corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="입력 변수 간 상관관계 히트맵"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    if dropped_features:
        st.warning(
            f"상관계수 절댓값이 **{threshold}** 이상인 다음 **{len(dropped_features)}개**의 변수가 "
            f"중복으로 간주되어 학습에서 제외됩니다:\n\n**{', '.join(dropped_features)}**"
        )
    else:
        st.success(f"상관계수 절댓값이 **{threshold}** 이상인 중복 변수가 없습니다.")
        
    st.info("이곳에서 설정한 임계값과 제거된 변수 목록은 '머신러닝 모델 제작' 화면에 자동으로 반영됩니다.")
    st.divider()
                           
    show_record_editor(record, list(raw_df.columns))
    st.caption(
        f"데이터 출처: [{record.get('data_source', DATA_SOURCE)}]"
        f"({record.get('data_url', DATA_URL)})"
    )


def main():
    st.title("데이터 분석 및 전처리")
    if "record" not in st.session_state:
        st.session_state["record"] = load_record()
    record = st.session_state["record"]

    try:
        with st.spinner("JM1 데이터셋을 불러오는 중입니다..."):
            raw_df, cleaned_df, feature_columns, duplicate_count, invalid_target_count = load_app_data()
    except Exception as error:
        st.error(
            "데이터를 불러오거나 전처리하지 못했습니다. 인터넷 연결과 데이터 주소, "
            f"CSV 형식을 확인하세요. 상세 오류: {error}"
        )
        st.info("데이터 주소: " + DATA_URL)
        return

    page_data_analysis(
        raw_df, cleaned_df, feature_columns, duplicate_count,
        invalid_target_count, record
    )
    st.divider()
    st.caption("참고: 입력 변수의 결측값 처리는 모델 학습 과정에서 학습 데이터만 이용해 수행합니다.")


if __name__ == "__main__":
    main()
