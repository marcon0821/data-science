import streamlit as st

from common import load_app_data, train_and_evaluate, evaluate_at_threshold, drop_high_correlation_features

st.set_page_config(
    page_title="머신러닝 모델 제작 | 소프트웨어 결함 예측",
    page_icon="🧠",
    layout="wide",
)


def page_model(cleaned_df, feature_columns):
    st.header("2. 머신러닝 모델 제작")
    st.write("분류 모델을 선택하고 학습 데이터 80%, 테스트 데이터 20%로 학습합니다.")

    selected_model = st.selectbox(
        "사용할 분류 모델",
        ["로지스틱 회귀", "랜덤 포레스트"],
        help="로지스틱 회귀는 선형 분류 모델이고, 랜덤 포레스트는 여러 결정 트리를 결합한 모델입니다.",
    )

    st.markdown("**학습 설정**")
    setting_cols = st.columns(3)
    setting_cols[0].metric("학습 데이터", "80%")
    setting_cols[1].metric("테스트 데이터", "20%")
    setting_cols[2].metric("불균형 처리", "class_weight")
    st.caption(
        "두 모델 모두 class_weight='balanced'를 사용해 학습 데이터의 클래스 개수 차이를 반영합니다. "
        "데이터 분할에도 stratify를 적용해 학습/테스트 데이터의 결함 비율이 비슷하게 유지되도록 합니다."
    )
    # ... 기존 코드(st.caption("두 모델 모두...")) 아래에 다음을 추가/수정합니다.

    # 데이터 전처리 페이지에서 설정한 임계값을 불러와 변수 축소 적용
    threshold = st.session_state.get("corr_threshold", 0.90)
    reduced_features, dropped_features, _ = drop_high_correlation_features(
        cleaned_df, feature_columns, threshold=threshold
    )

    if dropped_features:
        st.info(
            f"🧹 데이터 전처리 페이지의 설정(임계값 {threshold})에 따라 "
            f"중복 변수 {len(dropped_features)}개가 학습에서 제외됩니다: {', '.join(dropped_features)}"
        )

    if st.button("모델 학습 시작", type="primary", use_container_width=True):
        try:
            with st.spinner("데이터를 나누고 모델을 학습하는 중입니다..."):
                # 기존의 feature_columns 대신 축소된 reduced_features를 넣어 학습합니다.
                result = train_and_evaluate(cleaned_df, reduced_features, selected_model)
            st.session_state["model_bundle"] = result
            st.session_state["decision_threshold"] = 0.5
            st.success(f"{selected_model} 학습이 완료되었습니다.")
        except Exception as error:
            st.error(
                "모델을 학습하지 못했습니다. 데이터의 결함 있음/없음 개수와 "
                f"입력 변수 상태를 확인하세요. 상세 오류: {error}"
            )

    bundle = st.session_state.get("model_bundle")
    if bundle:
        st.divider()
        st.subheader("학습 결과")
        if bundle["model_name"] != selected_model:
            st.warning(
                f"현재 선택은 {selected_model}이지만, 저장된 학습 결과는 "
                f"{bundle['model_name']}입니다. 새 모델을 학습하려면 위 버튼을 누르세요."
            )
        result_cols = st.columns(3)
        result_cols[0].metric("사용 모델", bundle["model_name"])
        result_cols[1].metric("학습 데이터", f"{bundle['train_rows']:,}개")
        result_cols[2].metric("테스트 데이터", f"{bundle['test_rows']:,}개")
        st.write(f"학습에 사용된 입력 변수: {len(bundle['feature_columns'])}개")
        if bundle["dropped_empty_columns"]:
            st.warning(
                "모든 값이 결측이라 학습에서 제외된 열: "
                + ", ".join(bundle["dropped_empty_columns"])
            )

        st.subheader("예측 임계값 설정")
        st.write(
            "모델은 각 모듈의 결함 발생 확률을 계산한 뒤, 설정한 임계값과 비교해 "
            "'결함 있음' 또는 '결함 없음'으로 분류합니다."
        )
        st.session_state.setdefault("decision_threshold", 0.5)
        threshold = st.slider(
            "결함 있음으로 판단할 최소 확률(임계값)",
            min_value=0.05,
            max_value=0.95,
            step=0.05,
            format="%.2f",
            key="decision_threshold",
            help="예를 들어 0.30이면 결함 확률이 30% 이상일 때 결함 있음으로 분류합니다.",
        )
        st.info(
            f"현재 임계값은 {threshold:.2f}({threshold * 100:.0f}%)입니다. "
            f"결함 확률이 {threshold * 100:.0f}% 이상이면 '결함 있음'으로 판단합니다. "
            "임계값을 낮추면 결함을 더 많이 찾아낼 수 있지만 정상 모듈을 결함으로 오판하는 경우도 늘 수 있습니다."
        )
        preview = evaluate_at_threshold(bundle, threshold)
        preview_cols = st.columns(3)
        preview_cols[0].metric("현재 임계값 Recall", f"{preview['recall'] * 100:.2f}%")
        preview_cols[1].metric("현재 임계값 Precision", f"{preview['precision'] * 100:.2f}%")
        preview_cols[2].metric("현재 임계값 F1-score", f"{preview['f1'] * 100:.2f}%")
        st.caption(
            "임계값 변경은 학습된 모델을 다시 학습하는 것이 아니라, 같은 예측 확률에 적용하는 "
            "분류 기준만 바꾸는 기능입니다. 상세 비교는 '모델 성능 평가' 화면에서 확인하세요."
        )
        st.info(
            "학습이 끝났습니다. 왼쪽 메뉴의 '모델 성능 평가'에서 임계값별 지표와 혼동 행렬을, "
            "'사용자 코드 결함 예측'에서 직접 입력한 값의 예측 결과를 확인할 수 있습니다."
        )
    else:
        st.info("아직 학습된 모델이 없습니다. 모델을 선택한 뒤 '모델 학습 시작'을 누르세요.")


def main():
    try:
        with st.spinner("JM1 데이터셋을 불러오는 중입니다..."):
            _, cleaned_df, feature_columns, _, _ = load_app_data()
    except Exception as error:
        st.error(f"데이터를 불러오지 못했습니다. 상세 오류: {error}")
        return
    page_model(cleaned_df, feature_columns)


if __name__ == "__main__":
    main()
