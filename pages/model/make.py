import streamlit as st

from common import load_app_data, train_and_evaluate, drop_high_correlation_features, load_record

st.set_page_config(
    page_title="머신러닝 모델 제작 | 소프트웨어 결함 예측",
    page_icon="🧠",
    layout="wide",
)


def page_model(cleaned_df, feature_columns, record):
    st.header("2. 머신러닝 모델 제작")
    st.write("분류 모델을 선택하고, 학습할 변수(특성)와 파라미터를 직접 지정하여 모델을 훈련합니다.")

    # ------------------------------------------------------------------
    # 1. 모델 및 사용자 정의 변수 선택 (Feature Selection)
    # ------------------------------------------------------------------
    st.subheader("⚙️ 모델 및 속성 설정")
    
    selected_model = st.selectbox(
        "사용할 분류 모델",
        ["로지스틱 회귀", "랜덤 포레스트"],
        help="로지스틱 회귀는 선형 분류 모델이고, 랜덤 포레스트는 여러 결정 트리를 결합한 모델입니다.",
    )
    
    meanings = record.get("variable_meanings", {})
    selected_features = st.multiselect(
        "학습에 사용할 속성을 선택해 주세요 (최소 1개 이상)",
        options=feature_columns,
        default=feature_columns,
        format_func=lambda x: f"{x} ({meanings.get(x, '설명 없음')})",
        help="특정 지표(예: 코드 규모 loc, 복잡도 v(g))만 남겨서 모델의 성능 변화를 확인할 수 있습니다."
    )

    if not selected_features:
        st.warning("⚠️ 학습을 진행하려면 최소 1개 이상의 속성을 선택해야 합니다.")
        st.stop()

    # 데이터 전처리 페이지의 상관계수 설정 연동 (선택한 변수 안에서만 필터링)
    threshold_corr = st.session_state.get("corr_threshold", 0.90)
    reduced_features, dropped_features, _ = drop_high_correlation_features(
        cleaned_df, selected_features, threshold=threshold_corr
    )

    if dropped_features:
        st.info(
            f"🧹 데이터 전처리 페이지의 설정(상관계수 임계값 {threshold_corr})에 따라 "
            f"사용자가 선택한 변수 중 **{len(dropped_features)}개**가 중복으로 간주되어 제외됩니다: {', '.join(dropped_features)}"
        )

    # ------------------------------------------------------------------
    # 2. 클래스 가중치(class_weight) 토글
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("⚖️ 하이퍼파라미터 튜닝")
    use_class_weight = st.checkbox(
        "클래스 가중치 보정 켜기 (class_weight='balanced')", 
        value=True,
        help="켜면 결함(True)과 정상(False) 데이터의 비율 차이를 모델이 스스로 보정합니다. 끄면 정상 코드만 잘 맞추고 결함을 놓치는 현상을 관찰할 수 있습니다."
    )

    st.markdown("**학습 데이터 설정**")
    setting_cols = st.columns(3)
    setting_cols[0].metric("학습 데이터 비율", "80%")
    setting_cols[1].metric("테스트 데이터 비율", "20%")
    setting_cols[2].metric("가중치 보정 여부", "적용됨" if use_class_weight else "미적용")

    if st.button("🚀 모델 학습 시작", type="primary", use_container_width=True):
        try:
            with st.spinner(f"{selected_model} 모델을 학습하는 중입니다..."):
                # train_and_evaluate 호출 시 선택된 변수와 가중치 옵션 전달
                result = train_and_evaluate(
                    cleaned_df, 
                    reduced_features, 
                    selected_model, 
                    use_class_weight=use_class_weight
                )
            st.session_state["model_bundle"] = result
            st.session_state["decision_threshold"] = 0.5
            st.success(f"[{selected_model}] 학습이 완료되었습니다! '모델 성능 평가' 메뉴에서 결과를 확인하세요.")
        except Exception as error:
            st.error(
                "모델을 학습하지 못했습니다. 데이터 상태와 변수를 확인하세요. "
                f"상세 오류: {error}"
            )

    bundle = st.session_state.get("model_bundle")
    if bundle:
        st.divider()
        st.subheader("학습 결과 요약")
        if bundle["model_name"] != selected_model:
            st.warning(
                f"현재 위에서 선택된 모델은 '{selected_model}'이지만, 화면에 표시된 결과는 이전에 학습한 "
                f"'{bundle['model_name']}'입니다. 새 설정을 적용하려면 '모델 학습 시작' 버튼을 누르세요."
            )
        result_cols = st.columns(4)
        result_cols[0].metric("사용 모델", bundle["model_name"])
        result_cols[1].metric("학습 데이터", f"{bundle['train_rows']:,}개")
        result_cols[2].metric("테스트 데이터", f"{bundle['test_rows']:,}개")
        result_cols[3].metric("최종 사용 변수", f"{len(bundle['feature_columns'])}개")
        
        with st.expander("🔍 학습에 사용된 최종 변수 목록 보기"):
            st.write(", ".join(bundle['feature_columns']))
            
        if bundle["dropped_empty_columns"]:
            st.warning(
                "모든 값이 결측이라 학습에서 자동으로 제외된 열: "
                + ", ".join(bundle["dropped_empty_columns"])
            )


def main():
    if "record" not in st.session_state:
        st.session_state["record"] = load_record()
    record = st.session_state["record"]

    try:
        with st.spinner("JM1 데이터셋을 불러오는 중입니다..."):
            _, cleaned_df, feature_columns, _, _ = load_app_data()
    except Exception as error:
        st.error(f"데이터를 불러오지 못했습니다. 상세 오류: {error}")
        return
    page_model(cleaned_df, feature_columns, record)


if __name__ == "__main__":
    main()
