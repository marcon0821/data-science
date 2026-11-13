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

    # ---------------------------------------------------------
    # 신규: 고급 전처리 옵션 체크박스
    # ---------------------------------------------------------
    st.divider()
    st.subheader("🛠️ 고급 전처리 옵션 (성능 튜닝)")
    st.write("아래 옵션들을 껐다 켰다 하면서, '모델 성능 평가' 페이지에서 재현율과 정확도가 어떻게 변하는지 비교해 보세요!")
    
    prep_cols = st.columns(3)
    with prep_cols[0]:
        use_log = st.checkbox("📈 로그 변환 (Log Transform)", 
                              help="한쪽으로 쏠린 데이터(예: 코드 줄 수)를 둥글게 만들어 패턴을 쉽게 찾게 합니다.")
    with prep_cols[1]:
        use_cap = st.checkbox("✂️ 이상치 제어 (Capping)", 
                              help="상위 1%의 극단적인 스파게티 코드를 잘라내어 모델의 혼란을 막습니다.")
    with prep_cols[2]:
        use_smote = st.checkbox("⚖️ 데이터 균형 (SMOTE)", 
                                help="부족한 결함 데이터를 가상으로 생성하여 결함:정상 비율을 5:5로 맞춥니다.")

    st.divider()
    st.markdown("**학습 설정**")
    setting_cols = st.columns(3)
    setting_cols[0].metric("학습 데이터", "80%")
    setting_cols[1].metric("테스트 데이터", "20%")
    setting_cols[2].metric("불균형 처리", "SMOTE" if use_smote else "class_weight")

    threshold_corr = st.session_state.get("corr_threshold", 0.90)
    reduced_features, dropped_features, _ = drop_high_correlation_features(
        cleaned_df, feature_columns, threshold=threshold_corr
    )

    if dropped_features:
        st.info(
            f"🧹 데이터 전처리 페이지의 설정(상관계수 {threshold_corr})에 따라 "
            f"중복 변수 {len(dropped_features)}개가 제외됩니다: {', '.join(dropped_features)}"
        )

    if st.button("모델 학습 시작", type="primary", use_container_width=True):
        try:
            with st.spinner("데이터를 나누고 전처리를 적용하여 모델을 학습하는 중입니다..."):
                result = train_and_evaluate(
                    cleaned_df, reduced_features, selected_model,
                    use_log=use_log, use_cap=use_cap, use_smote=use_smote
                )
            st.session_state["model_bundle"] = result
            st.session_state["decision_threshold"] = 0.5
            st.success(f"[{selected_model}] 학습이 완료되었습니다! '모델 성능 평가' 메뉴에서 결과를 확인하세요.")
        except ImportError as ie:
            st.error(str(ie))
        except Exception as error:
            st.error(f"모델 학습 실패. 상세 오류: {error}")

    # 기존 학습 결과 UI는 그대로 유지
    bundle = st.session_state.get("model_bundle")
    if bundle:
        st.divider()
        st.subheader("학습 결과 요약")
        result_cols = st.columns(4)
        result_cols[0].metric("사용 모델", bundle["model_name"])
        result_cols[1].metric("학습 데이터", f"{bundle['train_rows']:,}개")
        result_cols[2].metric("테스트 데이터", f"{bundle['test_rows']:,}개")
        result_cols[3].metric("사용된 변수", f"{len(bundle['feature_columns'])}개")

def main():
    try:
        with st.spinner("JM1 데이터셋을 불러오는 중입니다..."):
            _, cleaned_df, feature_columns, _, _ = load_app_data()
    except Exception as error:
        st.error(f"데이터 로드 실패: {error}")
        return
    page_model(cleaned_df, feature_columns)

if __name__ == "__main__":
    main()
