import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import evaluate_at_threshold

st.set_page_config(
    page_title="모델 성능 평가 | 소프트웨어 결함 예측",
    page_icon="📊",
    layout="wide",
)


def page_evaluation():
    st.header("3. 모델 성능 평가")
    bundle = st.session_state.get("model_bundle")
    if not bundle:
        st.warning("먼저 '머신러닝 모델 제작' 화면에서 모델을 학습해 주세요.")
        return

    st.write(f"평가 모델: **{bundle['model_name']}** · 테스트 데이터 {bundle['test_rows']:,}개")
    st.subheader("임계값 설정")
    st.markdown(
        "**임계값(Threshold)**은 모델이 계산한 결함 발생 확률을 실제 분류 결과로 바꾸는 기준이야. "
        "예를 들어 임계값이 0.30이면 결함 발생 확률이 30% 이상인 모듈을 '결함 있음'으로 분류하고, "
        "30% 미만이면 '결함 없음'으로 분류해."
    )
    st.caption(
        "임계값을 낮추면 결함을 놓치는 경우(FN)가 줄어 재현율이 높아지는 경향이 있지만, "
        "정상 모듈을 결함으로 잘못 판단하는 경우(FP)가 늘어 정밀도가 낮아질 수 있어."
    )
    st.session_state.setdefault("decision_threshold", 0.5)
    threshold = st.slider(
        "평가에 사용할 임계값",
        min_value=0.05,
        max_value=0.95,
        step=0.05,
        format="%.2f",
        key="decision_threshold",
    )
    st.session_state["decision_threshold"] = threshold
    current = evaluate_at_threshold(bundle, threshold)
    metrics = current

    metric_columns = st.columns(4)
    metric_columns[0].metric("정확도 (Accuracy)", f"{metrics['accuracy'] * 100:.2f}%")
    metric_columns[1].metric("정밀도 (Precision)", f"{metrics['precision'] * 100:.2f}%")
    metric_columns[2].metric("재현율 (Recall)", f"{metrics['recall'] * 100:.2f}%")
    metric_columns[3].metric("F1-score", f"{metrics['f1'] * 100:.2f}%")
    st.caption(f"위 지표는 현재 임계값 {threshold:.2f}을 적용한 결과입니다.")

    st.subheader("혼동 행렬 (Confusion Matrix)")
    matrix = current["confusion_matrix"]
    labels = ["실제: 결함 없음", "실제: 결함 있음"]
    x_labels = ["예측: 결함 없음", "예측: 결함 있음"]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=labels,
            colorscale="Blues",
            showscale=True,
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 18},
            hovertemplate="%{y}<br>%{x}<br>개수: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="현재 임계값에서 실제 분류와 모델 예측 비교", xaxis_title="모델 예측 결과", yaxis_title="실제 정답")
    st.plotly_chart(fig, use_container_width=True)

    tn, fp = matrix[0]
    fn, tp = matrix[1]
    detail_columns = st.columns(4)
    detail_columns[0].metric("TN · 정상으로 예측", f"{tn:,}개")
    detail_columns[1].metric("FP · 결함으로 잘못 예측", f"{fp:,}개")
    detail_columns[2].metric("FN · 결함을 놓침", f"{fn:,}개")
    detail_columns[3].metric("TP · 결함을 찾아냄", f"{tp:,}개")

    st.subheader("임계값별 성능 비교")
    st.write(
        "같은 테스트 데이터의 예측 확률에 여러 임계값을 적용한 결과입니다. "
        "임계값만 바꾸어 계산하므로 모델을 매번 다시 학습하지 않습니다."
    )
    threshold_values = [round(value, 2) for value in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]]
    comparison_rows = []
    for value in threshold_values:
        result = evaluate_at_threshold(bundle, value)
        comparison_rows.append({
            "임계값": value,
            "정확도(%)": round(result["accuracy"] * 100, 2),
            "정밀도(%)": round(result["precision"] * 100, 2),
            "재현율(%)": round(result["recall"] * 100, 2),
            "F1-score(%)": round(result["f1"] * 100, 2),
        })
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    chart_df = comparison_df.melt(
        id_vars="임계값", var_name="평가 지표", value_name="점수(%)"
    )
    comparison_fig = px.line(
        chart_df,
        x="임계값",
        y="점수(%)",
        color="평가 지표",
        markers=True,
        title="임계값 변화에 따른 성능 지표",
    )
    comparison_fig.update_layout(yaxis=dict(range=[0, 100]), xaxis_title="임계값", yaxis_title="점수(%)")
    st.plotly_chart(comparison_fig, use_container_width=True)
    st.info(
        f"현재 선택한 임계값은 {threshold:.2f}입니다. 표와 그래프를 비교해 결함을 놓치는 정도(재현율)와 "
        "정상 모듈을 잘못 경고하는 정도(정밀도) 사이의 균형을 확인하세요. 최종 임계값은 "
        "테스트 결과를 보고 무조건 가장 높은 점수로 고르기보다, 별도 검증 데이터에서 결정하는 것이 바람직합니다."
    )

    st.subheader("평가 지표의 의미")
    explanation = pd.DataFrame(
        [
            ["Accuracy (정확도)", "전체 테스트 데이터 중 맞게 예측한 비율"],
            ["Precision (정밀도)", "결함이 있다고 예측한 데이터 중 실제 결함인 비율"],
            ["Recall (재현율)", "실제 결함 데이터 중 모델이 결함이라고 찾아낸 비율"],
            ["F1-score", "정밀도와 재현율을 함께 고려한 점수"],
        ],
        columns=["지표", "의미"],
    )
    st.dataframe(explanation, use_container_width=True, hide_index=True)

    st.info(
        f"실제 결함 모듈 중 모델이 찾아낸 비율(재현율)은 {metrics['recall'] * 100:.2f}%입니다. "
        f"테스트 데이터의 결함 모듈 {fn + tp:,}개 중 {tp:,}개를 찾아냈습니다."
    )

    st.subheader("결과 해석 시 주의할 점")
    st.markdown(
        "- 결함이 없는 데이터가 훨씬 많으면 모든 데이터를 '결함 없음'으로 예측해도 정확도가 높게 나올 수 있습니다. "
        "따라서 재현율과 F1-score도 함께 확인해야 합니다.\n"
        "- 재현율을 높이면 정상 모듈을 결함으로 잘못 판단하는 경우(FP)가 늘어날 수 있습니다. "
        "실제 사용 목적에 따라 FP와 FN 중 어느 오류가 더 큰 문제인지 고려해야 합니다.\n"
        "- 이 모델은 JM1 데이터의 통계적 특성을 학습한 결과입니다. 다른 언어·프로젝트·개발 환경에서는 "
        "성능이 달라질 수 있으며, 실제 소프트웨어 전체에 그대로 적용할 수는 없습니다.\n"
        "- class_weight를 적용했으므로, 예측 확률을 실제 서비스의 결함 발생률처럼 해석해서는 안 됩니다."
    )


if __name__ == "__main__":
    page_evaluation()
