"""모델 성능 평가 페이지.

main.py의 st.navigation에서 호출되며, common.py의 evaluate_at_threshold와
make.py가 저장한 session_state['model_bundle'] 구조를 사용한다.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from common import evaluate_at_threshold, load_app_data, TARGET


st.set_page_config(
    page_title="모델 성능 평가 | 소프트웨어 결함 예측",
    page_icon="📊",
    layout="wide",
)


def _get_threshold_state():
    """모델 제작 페이지와 공유하는 임계값을 슬라이더 생성 전에 준비한다."""
    key = "decision_threshold"

    if key not in st.session_state:
        st.session_state[key] = 0.5
        return

    try:
        value = float(st.session_state[key])
    except (TypeError, ValueError):
        value = 0.5

    if not 0.05 <= value <= 0.95:
        value = min(0.95, max(0.05, value))

    st.session_state[key] = round(value / 0.05) * 0.05


def _make_comparison_dataframe(bundle):
    """여러 임계값에서의 성능을 common.py 함수로 계산한다."""
    threshold_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    rows = []

    for value in threshold_values:
        result = evaluate_at_threshold(bundle, value)
        rows.append(
            {
                "임계값": value,
                "정확도(%)": round(result["accuracy"] * 100, 2),
                "정밀도(%)": round(result["precision"] * 100, 2),
                "재현율(%)": round(result["recall"] * 100, 2),
                "F1-score(%)": round(result["f1"] * 100, 2),
            }
        )

    return pd.DataFrame(rows)


def page_evaluation():
    st.header("3. 모델 성능 평가")

    bundle = st.session_state.get("model_bundle")
    if not bundle:
        st.warning("먼저 '머신러닝 모델 제작' 화면에서 모델을 학습해 주세요.")
        return

    required_keys = {"model_name", "test_rows", "y_test", "defect_probabilities"}
    if required_keys.difference(bundle.keys()):
        st.error("저장된 모델 정보의 형식이 맞지 않습니다. 모델을 다시 학습해 주세요.")
        return

    st.write(f"평가 모델: **{bundle['model_name']}** · 테스트 데이터 {int(bundle['test_rows']):,}개")

    # ------------------------------------------------------------------
    # 1. 임계값 설정
    # ------------------------------------------------------------------
    st.subheader("임계값 설정")
    st.markdown(
        "**임계값(Threshold)**은 모델이 계산한 결함 발생 확률을 '결함 있음' 또는 '결함 없음'으로 바꾸는 기준값입니다."
    )
    
    _get_threshold_state()
    threshold = st.slider(
        "평가에 사용할 임계값",
        min_value=0.05, max_value=0.95, step=0.05, format="%.2f",
        key="decision_threshold",
    )

    try:
        current = evaluate_at_threshold(bundle, float(threshold))
    except Exception as error:
        st.error(f"성능을 계산하지 못했습니다. 상세 오류: {error}")
        return

    # ------------------------------------------------------------------
    # [신규 추가] 2. 베이스라인(바보 모델)과의 성능 비교
    # ------------------------------------------------------------------
    st.subheader("📋 머신러닝 vs 베이스라인(바보 모델) 성능 비교")
    st.write(
        "베이스라인(바보 모델)은 코드의 특성을 전혀 보지 않고, 확률적으로 가장 많은 "
        "**'무조건 결함 없음(0)'**으로만 찍는 모델입니다. 머신러닝이 왜 필요한지 증명합니다."
    )

    y_true = bundle["y_test"]
    dummy_preds = [0] * len(y_true)  # 모두 정상(0)으로 예측
    
    dummy_acc = accuracy_score(y_true, dummy_preds)
    dummy_prec = precision_score(y_true, dummy_preds, pos_label=1, zero_division=0)
    dummy_rec = recall_score(y_true, dummy_preds, pos_label=1, zero_division=0)
    dummy_f1 = f1_score(y_true, dummy_preds, pos_label=1, zero_division=0)

    comp_df = pd.DataFrame(
        {
            "모델": [f"{bundle['model_name']} (현재 임계값 {threshold:.2f})", "베이스라인 (무조건 정상 예측)"],
            "정확도 (Accuracy)": [f"{current['accuracy']*100:.2f}%", f"{dummy_acc*100:.2f}%"],
            "재현율 (Recall)": [f"{current['recall']*100:.2f}%", f"{dummy_rec*100:.2f}%"],
            "정밀도 (Precision)": [f"{current['precision']*100:.2f}%", f"{dummy_prec*100:.2f}%"],
            "F1-score": [f"{current['f1']*100:.2f}%", f"{dummy_f1*100:.2f}%"],
        }
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.info(
        "💡 **분석 포인트:** 베이스라인 모델도 정상 데이터가 워낙 많아 정확도는 높게 나올 수 있습니다. "
        "하지만 결함을 찾아내는 능력(재현율)이 0%이므로 쓸모가 없습니다. "
        "머신러닝 모델은 재현율과 F1-score를 끌어올려 실제 결함을 탐지해냅니다."
    )

    # ------------------------------------------------------------------
    # 3. 혼동 행렬
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📊 혼동 행렬 (Confusion Matrix)")
    matrix = current["confusion_matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]

    y_labels = ["실제: 결함 없음", "실제: 결함 있음"]
    x_labels = ["예측: 결함 없음", "예측: 결함 있음"]
    confusion_fig = go.Figure(
        data=go.Heatmap(
            z=matrix, x=x_labels, y=y_labels, colorscale="Blues", showscale=True,
            text=matrix, texttemplate="%{text}", textfont={"size": 18},
            hovertemplate="%{y}<br>%{x}<br>개수: %{z}<extra></extra>",
        )
    )
    confusion_fig.update_layout(
        title="현재 임계값에서 실제 분류와 모델 예측 비교",
        xaxis_title="모델 예측 결과", yaxis_title="실제 정답",
    )
    st.plotly_chart(confusion_fig, use_container_width=True)

    detail_columns = st.columns(4)
    detail_columns[0].metric("TN · 정상으로 예측", f"{tn:,}개")
    detail_columns[1].metric("FP · 결함으로 잘못 예측", f"{fp:,}개")
    detail_columns[2].metric("FN · 결함을 놓침", f"{fn:,}개")
    detail_columns[3].metric("TP · 결함을 찾아냄", f"{tp:,}개")

    # ------------------------------------------------------------------
    # [신규 추가] 4. 오류 분석 목록 (놓친 결함 vs 헛짚은 모듈)
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("🔍 오류 분석 목록 (놓친 결함 vs 헛짚은 모듈)")
    st.markdown(
        "단순히 점수만 보지 않고, **모델이 어떤 특성을 가진 코드를 주로 틀리는지** 실제 데이터를 확인하여 정성적으로 분석합니다. "
        "*(데이터가 많은 경우 앞의 20줄만 표시합니다.)*"
    )

    try:
        # 원래의 X_test 데이터를 복원하여 원본 특성값을 띄움
        _, cleaned_df, _, _, _ = load_app_data()
        X = cleaned_df[bundle["feature_columns"]]
        y = cleaned_df[TARGET].astype(int)
        
        # 학습 때와 완전히 동일한 조건으로 분할해 X_test 추출
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        df_error = X_test.copy()
        df_error["실제_결함"] = y_true
        df_error["모델_예측"] = current["predictions"]

        # 놓친 결함 (FN: 실제는 1인데 0으로 예측)
        df_fn = df_error[(df_error["실제_결함"] == 1) & (df_error["모델_예측"] == 0)]
        # 헛짚은 모듈 (FP: 실제는 0인데 1로 예측)
        df_fp = df_error[(df_error["실제_결함"] == 0) & (df_error["모델_예측"] == 1)]

        col_err1, col_err2 = st.columns(2)
        
        with col_err1:
            st.markdown(f"##### 🚨 놓친 결함 모듈 (FN: {len(df_fn)}개)")
            if len(df_fn) > 20:
                st.caption("⚠️ 20줄을 초과하여 상위 20개만 표시합니다.")
                st.dataframe(df_fn.head(20).drop(columns=["실제_결함", "모델_예측"]), use_container_width=True)
            else:
                st.dataframe(df_fn.drop(columns=["실제_결함", "모델_예측"]), use_container_width=True)
                
        with col_err2:
            st.markdown(f"##### ⚠️ 헛짚은 정상 모듈 (FP: {len(df_fp)}개)")
            if len(df_fp) > 20:
                st.caption("⚠️ 20줄을 초과하여 상위 20개만 표시합니다.")
                st.dataframe(df_fp.head(20).drop(columns=["실제_결함", "모델_예측"]), use_container_width=True)
            else:
                st.dataframe(df_fp.drop(columns=["실제_결함", "모델_예측"]), use_container_width=True)

        st.caption(
            "💡 **분석 팁:** 놓친 모듈(FN)의 `loc`(코드 길이)나 복잡도가 비정상적으로 낮거나 높은지 확인해 보세요. "
            "모델의 사각지대를 찾아내는 중요한 근거가 됩니다."
        )

    except Exception as e:
        st.warning(f"오류 분석 데이터를 불러오는 중 문제가 발생했습니다: {e}")

    # ------------------------------------------------------------------
    # 5. 임계값별 성능 비교
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("📈 임계값 변화에 따른 성능 트렌드")
    try:
        comparison_df = _make_comparison_dataframe(bundle)
    except Exception as error:
        st.error(f"성능을 비교하지 못했습니다. 상세 오류: {error}")
        return

    chart_df = comparison_df.melt(id_vars="임계값", var_name="평가 지표", value_name="점수(%)")
    comparison_fig = px.line(
        chart_df, x="임계값", y="점수(%)", color="평가 지표", markers=True,
        title="임계값 변화에 따른 성능 지표",
    )
    comparison_fig.update_layout(yaxis={"range": [0, 100]}, xaxis_title="임계값", yaxis_title="점수(%)")
    st.plotly_chart(comparison_fig, use_container_width=True)


def main():
    page_evaluation()

if __name__ == "__main__":
    main()
