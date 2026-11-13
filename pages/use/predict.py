"""사용자 입력을 이용한 소프트웨어 결함 예측 페이지."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import get_feature_types, load_app_data, load_record


st.set_page_config(
    page_title="사용자 코드 결함 예측 | 소프트웨어 결함 예측",
    page_icon="🔍",
    layout="wide",
)


# JM1 데이터의 특성을 이해하기 쉽도록 화면에서 그룹으로 나눈다.
FEATURE_GROUPS = {
    "코드 규모 및 구성": [
        "loc",
        "lOCode",
        "lOComment",
        "lOBlank",
        "locCodeAndComment",
    ],
    "복잡도 및 분기": [
        "v(g)",
        "ev(g)",
        "iv(g)",
        "branchCount",
    ],
    "연산자·피연산자 지표": [
        "n",
        "v",
        "l",
        "d",
        "i",
        "e",
        "b",
        "t",
        "uniq_Op",
        "uniq_Opnd",
        "total_Op",
        "total_Opnd",
    ],
}


# -----------------------------------------------------------------------------
# 입력 UI 보조 함수
# -----------------------------------------------------------------------------
def _input_key(column):
    """특성 이름을 Streamlit 위젯의 고유 키로 변환한다."""
    return f"predict_input_{column}"


def _reset_prediction_inputs(feature_columns):
    """입력 위젯 상태와 마지막 예측 결과를 초기화한다."""
    for column in feature_columns:
        st.session_state.pop(_input_key(column), None)
    st.session_state.pop("last_prediction", None)


def _safe_scalar(value):
    """NumPy 자료형 등을 JSON으로 저장할 수 있는 기본 자료형으로 바꾼다."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _prepare_threshold_state():
    """임계값 위젯을 만들기 전에 공유 세션 값을 유효한 범위로 맞춘다."""
    key = "decision_threshold"

    if key not in st.session_state:
        st.session_state[key] = 0.5
        return

    try:
        value = float(st.session_state[key])
    except (TypeError, ValueError):
        value = 0.5

    value = min(0.95, max(0.05, value))
    st.session_state[key] = round(round(value / 0.05) * 0.05, 2)


def _get_classifier(model):
    """Pipeline 안에 있는 실제 분류기를 가져온다."""
    if hasattr(model, "named_steps"):
        return model.named_steps.get("classifier", model)
    return model


def _get_defect_probability(model, input_df):
    """클래스 1(결함 있음)에 해당하는 예측 확률을 반환한다."""
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "현재 저장된 모델에서 predict_proba를 사용할 수 없습니다. "
            "모델 제작 화면에서 분류 모델을 다시 학습해 주세요."
        )

    probabilities = model.predict_proba(input_df)[0]
    classifier = _get_classifier(model)
    classes = list(classifier.classes_)

    defect_index = None
    for index, label in enumerate(classes):
        try:
            if int(label) == 1:
                defect_index = index
                break
        except (TypeError, ValueError):
            if str(label).strip().lower() in {"true", "1", "yes"}:
                defect_index = index
                break

    if defect_index is None:
        raise ValueError(
            "학습된 모델에서 결함 있음(1) 클래스를 찾지 못했습니다. "
            "모델 제작 화면에서 모델을 다시 학습해 주세요."
        )

    return float(probabilities[defect_index])


def _render_numeric_input(column, data, meaning):
    """숫자형 특성의 기본값과 JM1 데이터 범위를 표시한다."""
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    default_value = float(values.median()) if not values.empty else 0.0
    is_integer = (
        not values.empty
        and pd.api.types.is_integer_dtype(data[column])
    )
    step = 1.0 if is_integer else 0.1

    key = _input_key(column)
    help_text = f"JM1 데이터 범위: {values.min():.4f} ~ {values.max():.4f}" if not values.empty else "학습 데이터에 숫자값이 없습니다."

    return st.number_input(
        f"{column} · {meaning}",
        min_value=0.0,
        value=default_value,
        step=step,
        format="%.4f",
        key=key,
        help=help_text,
    )


def _render_categorical_input(column, data, meaning):
    """범주형 특성은 학습 데이터에 존재하는 선택지만 제공한다."""
    options = data[column].dropna().unique().tolist()
    options = sorted(options, key=lambda value: str(value))
    if not options:
        options = [""]

    mode_values = data[column].dropna().mode()
    default_value = mode_values.iloc[0] if not mode_values.empty else options[0]
    default_index = options.index(default_value) if default_value in options else 0

    key = _input_key(column)

    return st.selectbox(
        f"{column} · {meaning}",
        options=options,
        index=default_index,
        key=key,
        help="학습 데이터에 포함된 값 중에서 선택합니다.",
    )


def _render_input_form(feature_columns, usable_data, numeric_columns, record):
    """JM1 특성을 그룹별로 입력받고 제출 여부와 입력값을 반환한다."""
    meanings = record.get("variable_meanings", {})
    input_values = {}

    st.subheader("코드 특성값 입력")
    st.caption(
        "각 항목의 기본값은 JM1 데이터의 중앙값 또는 최빈값입니다. "
        "실제 소스 코드 자체가 아니라, 코드 분석 도구로 계산한 특성값을 입력해야 합니다."
    )

    grouped_columns = {}
    assigned = set()
    for group_name, preferred_columns in FEATURE_GROUPS.items():
        columns = [column for column in preferred_columns if column in feature_columns]
        if columns:
            grouped_columns[group_name] = columns
            assigned.update(columns)

    remaining_columns = [column for column in feature_columns if column not in assigned]
    if remaining_columns:
        grouped_columns["기타 특성"] = remaining_columns

    with st.form("prediction_input_form", clear_on_submit=False):
        for group_name, columns in grouped_columns.items():
            st.markdown(f"#### {group_name}")
            input_columns = st.columns(2)

            for index, column in enumerate(columns):
                meaning = meanings.get(column, "데이터셋 원본 변수")
                with input_columns[index % 2]:
                    if column in numeric_columns:
                        value = _render_numeric_input(column, usable_data, meaning)
                    else:
                        value = _render_categorical_input(column, usable_data, meaning)
                    input_values[column] = _safe_scalar(value)

        submitted = st.form_submit_button(
            "🔍 결함 예측하기",
            type="primary",
            use_container_width=True,
        )

    return submitted, input_values


# -----------------------------------------------------------------------------
# 결과 분석 및 시각화 함수
# -----------------------------------------------------------------------------
def _make_range_comparison(input_values, usable_data, bounds, numeric_columns, meanings):
    rows = []
    for column in numeric_columns:
        if column not in input_values:
            continue

        values = pd.to_numeric(usable_data[column], errors="coerce").dropna()
        if values.empty:
            continue

        value = float(input_values[column])
        median = float(values.median())
        bound = bounds.get(column, {})
        minimum = float(bound.get("min", values.min()))
        maximum = float(bound.get("max", values.max()))
        lower_fence = float(bound.get("lower_fence", minimum))
        upper_fence = float(bound.get("upper_fence", maximum))

        if value < minimum or value > maximum:
            status = "학습 데이터 범위 밖"
        elif value < lower_fence or value > upper_fence:
            status = "일반 범위에서 벗어남"
        else:
            status = "일반 범위"

        rows.append(
            {
                "특성": column,
                "의미": meanings.get(column, ""),
                "입력값": value,
                "JM1 데이터 중앙값": median,
                "학습 최솟값": minimum,
                "학습 최댓값": maximum,
                "범위 확인": status,
            }
        )
    return pd.DataFrame(rows)


def _render_range_analysis(input_values, usable_data, bundle, numeric_columns, record):
    st.subheader("입력값과 학습 데이터 비교")
    comparison = _make_range_comparison(
        input_values=input_values,
        usable_data=usable_data,
        bounds=bundle.get("bounds", {}),
        numeric_columns=numeric_columns,
        meanings=record.get("variable_meanings", {}),
    )

    if comparison.empty:
        st.info("비교할 숫자형 특성이 없습니다.")
        return

    outside = comparison[comparison["범위 확인"] == "학습 데이터 범위 밖"]
    unusual = comparison[comparison["범위 확인"] == "일반 범위에서 벗어남"]

    if not outside.empty:
        st.warning(
            "다음 입력값은 학습 데이터의 최솟값~최댓값을 벗어납니다: "
            + ", ".join(outside["특성"].tolist())
            + ". 학습 과정에서 충분히 관찰하지 못한 값이므로 예측 결과를 특히 신중하게 해석하세요."
        )
    elif not unusual.empty:
        st.info(
            "다음 입력값은 학습 데이터의 사분위 범위를 기준으로 일반적인 범위에서 벗어납니다: "
            + ", ".join(unusual["특성"].tolist())
            + ". 단순히 특이한 값이라는 뜻이며, 잘못된 입력이라고 확정할 수는 없습니다."
        )
    else:
        st.success("모든 숫자형 입력값이 학습 데이터의 최솟값~최댓값 범위 안에 있습니다.")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "입력값": st.column_config.NumberColumn(format="%.4f"),
            "JM1 데이터 중앙값": st.column_config.NumberColumn(format="%.4f"),
            "학습 최솟값": st.column_config.NumberColumn(format="%.4f"),
            "학습 최댓값": st.column_config.NumberColumn(format="%.4f"),
        },
    )


def _get_feature_importance(model, meanings):
    classifier = _get_classifier(model)
    preprocessor = model.named_steps.get("preprocessor") if hasattr(model, "named_steps") else None

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except (AttributeError, ValueError):
        feature_names = []

    values = None
    value_label = "중요도"
    directions = None

    if hasattr(classifier, "feature_importances_"):
        values = list(classifier.feature_importances_)
        value_label = "특성 중요도"
    elif hasattr(classifier, "coef_"):
        coefficients = classifier.coef_
        if getattr(coefficients, "ndim", 1) == 2:
            coefficients = coefficients[0] if coefficients.shape[0] == 1 else abs(coefficients).mean(axis=0)
        values = [float(abs(value)) for value in coefficients]
        if getattr(classifier.coef_, "ndim", 1) == 2 and classifier.coef_.shape[0] == 1:
            directions = ["결함 확률 증가 방향" if value > 0 else "결함 확률 감소 방향" for value in classifier.coef_[0]]
        value_label = "계수 절댓값"

    if values is None:
        return pd.DataFrame()

    if not feature_names or len(feature_names) != len(values):
        feature_names = [f"특성 {index + 1}" for index in range(len(values))]

    rows = []
    for index, (raw_name, importance) in enumerate(zip(feature_names, values)):
        clean_name = raw_name.split("__", 1)[-1]
        display_name = clean_name
        if clean_name in meanings:
            display_name = f"{clean_name} · {meanings[clean_name]}"

        row = {
            "특성": display_name,
            value_label: float(importance),
        }
        if directions is not None and index < len(directions):
            row["계수 방향"] = directions[index]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(value_label, ascending=False).head(10)


def _render_feature_importance(bundle, record):
    with st.expander("📊 모델이 전반적으로 중요하게 사용한 특성", expanded=False):
        importance_df = _get_feature_importance(
            bundle["model"],
            record.get("variable_meanings", {}),
        )

        if importance_df.empty:
            st.info("현재 모델에서 특성 중요도 또는 계수를 가져오지 못했습니다.")
            return

        value_column = importance_df.columns[1]
        chart_df = importance_df.sort_values(value_column, ascending=True)
        fig = go.Figure(
            go.Bar(
                x=chart_df[value_column],
                y=chart_df["특성"],
                orientation="h",
                text=chart_df[value_column].map(lambda value: f"{value:.4f}"),
                textposition="outside",
            )
        )
        fig.update_layout(
            title="모델에서 중요도가 높은 상위 10개 특성",
            xaxis_title=value_column,
            yaxis_title="",
            height=max(360, 34 * len(chart_df)),
            margin=dict(l=10, r=40, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        if bundle.get("model_name") == "랜덤 포레스트":
            st.caption("랜덤 포레스트의 특성 중요도는 여러 결정 트리의 분할에서 각 특성이 얼마나 활용되었는지를 요약한 값입니다.")
        else:
            st.caption("로지스틱 회귀의 계수 절댓값을 비교했습니다. 입력 변수가 표준화된 모델에서 계수 크기는 상대적인 영향력을 살펴보는 데 활용할 수 있습니다.")


def _render_prediction_result(bundle, usable_data, numeric_columns, record):
    """마지막으로 제출한 입력값의 예측 결과를 표시한다."""
    result = st.session_state.get("last_prediction")
    if not result:
        return

    input_values = result["input_values"]
    defect_probability = float(result["defect_probability"])
    threshold = float(st.session_state.get("decision_threshold", 0.5))
    prediction = 1 if defect_probability >= threshold else 0

    st.divider()
    st.subheader("🎯 인공지능 예측 결과")
    st.caption(f"사용 모델: {bundle.get('model_name', '알 수 없음')} · 마지막으로 제출한 입력값에 대한 분석 결과")

    result_left, result_right = st.columns([1, 1.5])
    
    with result_left:
        st.write(" ")
        if prediction == 1:
            st.error("### 🚨 결함 발생 가능성 높음")
            st.write("이 모듈은 코드 구조상 결함이 내재되어 있을 확률이 높습니다. 코드 검토(Code Review) 및 추가 테스트를 강력히 권장합니다.")
        else:
            st.success("### ✅ 결함 발생 가능성 낮음")
            st.write("모델이 결함이 없다고 판단했습니다. 다만 무결성을 완벽히 보장하는 것은 아니므로 기본 테스트는 진행해 주세요.")
            
        st.metric("결함 발생 예측 확률", f"{defect_probability * 100:.1f}%")
        st.caption(f"(현재 적용된 임계값 기준: {threshold*100:.0f}%)")

    with result_right:
        # 확률을 게이지 차트(Gauge Chart)로 예쁘게 표현
        gauge_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = defect_probability * 100,
            title = {'text': "결함 위험도 (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prediction == 1 else "darkgreen"},
                'steps': [
                    {'range': [0, threshold*100], 'color': "lightgreen"},
                    {'range': [threshold*100, 100], 'color': "lightpink"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        gauge_fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

    # 기존 기능 유지 영역 (이상치 경고, 입력값 표, JSON 다운로드 등)
    _render_range_analysis(
        input_values=input_values,
        usable_data=usable_data,
        bundle=bundle,
        numeric_columns=numeric_columns,
        record=record,
    )

    st.subheader("입력한 특성값 요약")
    input_table = pd.DataFrame(
        [
            {
                "특성": column,
                "우리말 뜻": record.get("variable_meanings", {}).get(column, ""),
                "입력값": value,
            }
            for column, value in input_values.items()
        ]
    )
    st.dataframe(input_table, use_container_width=True, hide_index=True)

    export_data = {
        "model_name": bundle.get("model_name"),
        "defect_probability": defect_probability,
        "decision_threshold": threshold,
        "prediction": "결함 있음" if prediction == 1 else "결함 없음",
        "input_values": input_values,
        "note": "모델의 통계적 예측이며 실제 결함 여부를 확정하지 않습니다.",
    }
    st.download_button(
        "⬇️ 예측 결과 JSON 다운로드",
        data=json.dumps(export_data, ensure_ascii=False, indent=4),
        file_name="software_defect_prediction.json",
        mime="application/json",
        use_container_width=True,
    )

    _render_feature_importance(bundle, record)

    if bundle.get("dropped_empty_columns"):
        st.caption("학습 데이터에 값이 전혀 없어 모델에서 제외된 특성: " + ", ".join(bundle["dropped_empty_columns"]))

    st.info(
        "중요한 한계: 이 프로그램은 JM1 데이터의 코드 통계 특성을 학습해 결함 가능성을 분류합니다. "
        "실제 버그가 있는 코드 줄이나 위치를 찾아내는 정적 분석 도구는 아닙니다."
    )


# -----------------------------------------------------------------------------
# 페이지 실행
# -----------------------------------------------------------------------------
def main():
    st.title("🔍 사용자 코드 결함 예측")
    st.write(
        "소프트웨어 모듈의 코드 특성값을 직접 입력하면, "
        "학습된 머신러닝 모델이 결함 발생 가능성을 예측합니다."
    )

    bundle = st.session_state.get("model_bundle")
    if not bundle:
        st.warning("먼저 '머신러닝 모델 제작' 화면에서 모델을 학습해 주세요.")
        st.info("모델 학습이 완료되면 이 화면에서 특성값을 입력하고 예측할 수 있습니다.")
        return

    required_keys = {"model", "feature_columns", "bounds"}
    missing_keys = required_keys.difference(bundle.keys())
    if missing_keys:
        st.error(
            "저장된 모델 정보가 현재 예측 화면과 호환되지 않습니다. "
            "모델 제작 화면에서 모델을 다시 학습해 주세요. "
            f"누락된 항목: {', '.join(sorted(missing_keys))}"
        )
        return

    try:
        with st.spinner("JM1 데이터와 변수 정보를 준비하는 중입니다..."):
            _, cleaned_df, _, _, _ = load_app_data()

        feature_columns = bundle["feature_columns"]
        missing_columns = [column for column in feature_columns if column not in cleaned_df.columns]
        if missing_columns:
            st.error("현재 데이터에서 학습 당시의 입력 변수를 찾지 못했습니다: " + ", ".join(missing_columns))
            return

        usable_data = cleaned_df[feature_columns].copy()
        numeric_columns, _ = get_feature_types(usable_data, feature_columns)

        if "record" not in st.session_state:
            st.session_state["record"] = load_record()
        record = st.session_state["record"]

        st.markdown(
            f"**학습 모델:** {bundle.get('model_name', '알 수 없음')}  "
            f"\n**학습에 사용된 특성:** {len(feature_columns)}개"
        )

        st.divider()
        st.subheader("분류 임계값 설정")
        _prepare_threshold_state()
        threshold = st.slider(
            "결함 있음으로 판단할 최소 확률",
            min_value=0.05,
            max_value=0.95,
            step=0.05,
            format="%.2f",
            key="decision_threshold",
            help="예를 들어 0.30이면 결함 발생 예측 확률이 30% 이상일 때 '결함 있음'으로 분류합니다.",
        )
        st.caption(
            f"현재 기준: 결함 확률이 **{threshold * 100:.0f}% 이상**이면 결함 있음으로 분류합니다. "
            "임계값을 낮추면 결함을 더 많이 찾아낼 수 있지만 정상 모듈을 결함으로 오판할 가능성도 커질 수 있습니다."
        )

        st.divider()
        submitted, input_values = _render_input_form(
            feature_columns=feature_columns,
            usable_data=usable_data,
            numeric_columns=numeric_columns,
            record=record,
        )

        reset_col, help_col = st.columns([1, 3])
        with reset_col:
            st.button(
                "입력값 초기화",
                on_click=_reset_prediction_inputs,
                args=(feature_columns,),
                use_container_width=True,
            )
        with help_col:
            st.caption("초기화하면 모든 항목이 JM1 데이터의 중앙값 또는 최빈값으로 돌아갑니다.")

        if submitted:
            try:
                input_df = pd.DataFrame([input_values], columns=feature_columns)
                
                # 학습할 때 적용한 전처리(Capping, Log 변환) 똑같이 적용
                if bundle.get("use_cap") and bundle.get("cap_bounds"):
                    for col, upper in bundle["cap_bounds"].items():
                        if col in input_df.columns:
                            input_df[col] = input_df[col].clip(upper=upper)
                
                if bundle.get("use_log"):
                    input_df[numeric_columns] = np.log1p(input_df[numeric_columns].astype(float))

                probability = _get_defect_probability(bundle["model"], input_df)
                st.session_state["last_prediction"] = {
                    "input_values": input_values,
                    "defect_probability": probability,
                }
            except Exception as error:
                st.error(f"예측을 실행하지 못했습니다. 입력값과 모델을 확인하세요. 상세 오류: {error}")

        _render_prediction_result(
            bundle=bundle,
            usable_data=usable_data,
            numeric_columns=numeric_columns,
            record=record,
        )

    except Exception as error:
        st.error(f"예측 화면을 준비하지 못했습니다. 상세 오류: {error}")


if __name__ == "__main__":
    main()
