"""모델 성능 평가 페이지.

main.py의 st.navigation에서 호출되며, common.py의 evaluate_at_threshold와
make.py가 저장한 session_state['model_bundle'] 구조를 사용한다.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import evaluate_at_threshold


# 페이지 기본 설정은 프로젝트의 다른 페이지와 같은 형식으로 유지한다.
st.set_page_config(
    page_title="모델 성능 평가 | 소프트웨어 결함 예측",
    page_icon="📊",
    layout="wide",
)


def _get_threshold_state():
    """모델 제작 페이지와 공유하는 임계값을 슬라이더 생성 전에 준비한다.

    중요: 슬라이더를 생성한 뒤에는 같은 key의 session_state를 직접 수정하지 않는다.
    Streamlit이 슬라이더 값을 자동으로 session_state에 반영한다.
    """
    key = "decision_threshold"

    if key not in st.session_state:
        st.session_state[key] = 0.5
        return

    # 이전 세션 값이 비정상적이거나 현재 슬라이더 범위 밖이면 위젯 생성 전에 보정한다.
    try:
        value = float(st.session_state[key])
    except (TypeError, ValueError):
        value = 0.5

    if not 0.05 <= value <= 0.95:
        value = min(0.95, max(0.05, value))

    # 이 시점에는 아직 rating 페이지의 해당 슬라이더가 만들어지지 않았다.
    st.session_state[key] = round(value / 0.05) * 0.05


def _show_metric_cards(metrics):
    """현재 임계값의 네 가지 평가 지표를 표시한다."""
    columns = st.columns(4)
    columns[0].metric("정확도 (Accuracy)", f"{metrics['accuracy'] * 100:.2f}%")
    columns[1].metric("정밀도 (Precision)", f"{metrics['precision'] * 100:.2f}%")
    columns[2].metric("재현율 (Recall)", f"{metrics['recall'] * 100:.2f}%")
    columns[3].metric("F1-score", f"{metrics['f1'] * 100:.2f}%")


def _make_comparison_dataframe(bundle):
    """여러 임계값에서의 성능을 common.py 함수로 계산한다."""
    threshold_values = [
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90,
    ]
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
    """테스트 데이터 성능, 혼동 행렬, 임계값별 성능을 보여준다."""
    st.header("3. 모델 성능 평가")

    # make.py에서 학습한 모델 번들을 세션 상태에서 가져온다.
    bundle = st.session_state.get("model_bundle")
    if not bundle:
        st.warning("먼저 '머신러닝 모델 제작' 화면에서 모델을 학습해 주세요.")
        return

    # common.py의 train_and_evaluate()가 반환하는 키와 호환되는지 확인한다.
    required_keys = {
        "model_name",
        "test_rows",
        "y_test",
        "defect_probabilities",
    }
    missing_keys = required_keys.difference(bundle.keys())
    if missing_keys:
        st.error(
            "저장된 모델 정보의 형식이 현재 평가 페이지와 맞지 않습니다. "
            "머신러닝 모델 제작 화면에서 모델을 다시 학습해 주세요. "
            f"누락된 항목: {', '.join(sorted(missing_keys))}"
        )
        return

    st.write(
        f"평가 모델: **{bundle['model_name']}** · "
        f"테스트 데이터 {int(bundle['test_rows']):,}개"
    )

    # ------------------------------------------------------------------
    # 1. 임계값 설정
    # ------------------------------------------------------------------
    st.subheader("임계값 설정")
    st.markdown(
        "**임계값(Threshold)**은 모델이 계산한 결함 발생 확률을 "
        "'결함 있음' 또는 '결함 없음'으로 바꾸는 기준값입니다.\n\n"
        "예를 들어 임계값이 **0.30(30%)**이면 결함 발생 확률이 30% 이상인 "
        "모듈은 '결함 있음', 30% 미만인 모듈은 '결함 없음'으로 분류합니다."
    )
    st.caption(
        "임계값을 낮추면 결함을 놓치는 경우(FN)가 줄어 재현율이 높아지는 경향이 있습니다. "
        "반면 정상 모듈을 결함으로 잘못 판단하는 경우(FP)가 늘어 정밀도가 낮아질 수 있습니다."
    )

    _get_threshold_state()
    threshold = st.slider(
        "평가에 사용할 임계값",
        min_value=0.05,
        max_value=0.95,
        step=0.05,
        format="%.2f",
        key="decision_threshold",  # make.py와 공유하는 동일한 세션 키
        help="결함 발생 확률이 이 값 이상이면 '결함 있음'으로 분류합니다.",
    )

    # slider의 key 값을 아래에서 다시 대입하지 않는다.
    # Streamlit이 threshold 값을 session_state['decision_threshold']에 자동 저장한다.
    try:
        current = evaluate_at_threshold(bundle, float(threshold))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        st.error(
            "저장된 테스트 데이터로 성능을 계산하지 못했습니다. "
            "머신러닝 모델 제작 화면에서 모델을 다시 학습해 주세요. "
            f"상세 오류: {error}"
        )
        return

    _show_metric_cards(current)
    st.caption(f"위 지표는 현재 임계값 {threshold:.2f}을 적용한 결과입니다.")

    # ------------------------------------------------------------------
    # 2. 혼동 행렬
    # ------------------------------------------------------------------
    st.subheader("혼동 행렬 (Confusion Matrix)")
    matrix = current["confusion_matrix"]

    # common.py는 labels=[0, 1] 순서로 2x2 행렬을 반환한다.
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        st.error("혼동 행렬의 데이터 형식이 올바르지 않습니다. 모델을 다시 학습해 주세요.")
        return

    y_labels = ["실제: 결함 없음", "실제: 결함 있음"]
    x_labels = ["예측: 결함 없음", "예측: 결함 있음"]
    confusion_fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            showscale=True,
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 18},
            hovertemplate="%{y}<br>%{x}<br>개수: %{z}<extra></extra>",
        )
    )
    confusion_fig.update_layout(
        title="현재 임계값에서 실제 분류와 모델 예측 비교",
        xaxis_title="모델 예측 결과",
        yaxis_title="실제 정답",
    )
    st.plotly_chart(confusion_fig, use_container_width=True)

    tn, fp = matrix[0]
    fn, tp = matrix[1]
    detail_columns = st.columns(4)
    detail_columns[0].metric("TN · 정상으로 예측", f"{tn:,}개")
    detail_columns[1].metric("FP · 결함으로 잘못 예측", f"{fp:,}개")
    detail_columns[2].metric("FN · 결함을 놓침", f"{fn:,}개")
    detail_columns[3].metric("TP · 결함을 찾아냄", f"{tp:,}개")

    # ------------------------------------------------------------------
    # 3. 임계값별 성능 비교
    # ------------------------------------------------------------------
    st.subheader("임계값별 성능 비교")
    st.write(
        "같은 테스트 데이터의 예측 확률에 여러 임계값을 적용한 결과입니다. "
        "임계값만 바꾸어 계산하므로 모델을 매번 다시 학습하지 않습니다."
    )

    try:
        comparison_df = _make_comparison_dataframe(bundle)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        st.error(f"임계값별 성능을 비교하지 못했습니다. 상세 오류: {error}")
        return

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    chart_df = comparison_df.melt(
        id_vars="임계값",
        var_name="평가 지표",
        value_name="점수(%)",
    )
    comparison_fig = px.line(
        chart_df,
        x="임계값",
        y="점수(%)",
        color="평가 지표",
        markers=True,
        title="임계값 변화에 따른 성능 지표",
    )
    comparison_fig.update_layout(
        yaxis={"range": [0, 100]},
        xaxis_title="임계값",
        yaxis_title="점수(%)",
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

    st.info(
        f"현재 선택한 임계값은 {threshold:.2f}입니다. 표와 그래프를 비교해 "
        "결함을 놓치는 정도(재현율)와 정상 모듈을 잘못 경고하는 정도(정밀도) "
        "사이의 균형을 확인하세요. 최종 임계값은 테스트 결과에 과도하게 맞추기보다 "
        "별도의 검증 데이터에서 결정하는 것이 바람직합니다."
    )

    # ------------------------------------------------------------------
    # 4. 지표 설명 및 해석상 주의점
    # ------------------------------------------------------------------
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

    actual_defect_count = fn + tp
    st.info(
        f"실제 결함 모듈 중 모델이 찾아낸 비율(재현율)은 "
        f"{current['recall'] * 100:.2f}%입니다. 테스트 데이터의 결함 모듈 "
        f"{actual_defect_count:,}개 중 {tp:,}개를 찾아냈습니다."
    )

    st.subheader("결과 해석 시 주의할 점")
    st.markdown(
        "- 결함이 없는 데이터가 훨씬 많으면 모든 데이터를 '결함 없음'으로 예측해도 "
        "정확도가 높게 나올 수 있습니다. 따라서 재현율과 F1-score도 함께 확인해야 합니다.\n"
        "- 재현율을 높이면 정상 모듈을 결함으로 잘못 판단하는 경우(FP)가 늘어날 수 있습니다. "
        "실제 사용 목적에 따라 FP와 FN 중 어느 오류가 더 큰 문제인지 고려해야 합니다.\n"
        "- 이 모델은 JM1 데이터의 통계적 특성을 학습한 결과입니다. 다른 언어·프로젝트·개발 "
        "환경에서는 성능이 달라질 수 있으며, 실제 소프트웨어 전체에 그대로 적용할 수는 없습니다.\n"
        "- class_weight를 적용했으므로, 모델이 출력하는 확률을 실제 서비스의 결함 발생률처럼 "
        "해석해서는 안 됩니다."
    )


def main():
    page_evaluation()


if __name__ == "__main__":
    main()
