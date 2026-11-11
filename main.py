"""소프트웨어 결함 예측 프로그램

실행: streamlit run main.py
데이터: NASA PROMISE Software Defect Dataset - JM1.csv
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# 기본 설정
# -----------------------------------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/ApoorvaKrisna/"
    "NASA-promise-dataset-repository/refs/heads/main/jm1.csv"
)
DATA_SOURCE = "NASA PROMISE Software Defect Dataset (JM1.csv)"
TARGET = "defects"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "변수_뜻과_데이터_출처.json"

# 데이터셋에 실제로 존재하는 열의 우리말 설명
DEFAULT_MEANINGS = {
    "loc": "코드 전체 줄 수",
    "v(g)": "순환 복잡도",
    "ev(g)": "설계 복잡도",
    "iv(g)": "설계 구조의 복잡도 지표",
    "n": "연산자와 피연산자의 총 개수",
    "v": "코드의 크기를 나타내는 지표",
    "l": "프로그램의 이해 용이성 지표",
    "d": "프로그램의 난이도 지표",
    "i": "프로그램의 지능도 지표",
    "e": "프로그램 구현에 필요한 노력의 추정치",
    "b": "프로그램의 규모를 나타내는 지표",
    "t": "구현에 필요한 시간의 추정치",
    "lOCode": "코드 줄 수",
    "lOComment": "주석 줄 수",
    "lOBlank": "빈 줄 수",
    "locCodeAndComment": "코드와 주석이 함께 있는 줄 수",
    "uniq_Op": "서로 다른 연산자 개수",
    "uniq_Opnd": "서로 다른 피연산자 개수",
    "total_Op": "전체 연산자 개수",
    "total_Opnd": "전체 피연산자 개수",
    "branchCount": "분기문 개수",
    "defects": "결함 발생 여부 (True: 결함 있음, False: 결함 없음)",
}

st.set_page_config(
    page_title="소프트웨어 결함 예측 프로그램",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# 기록(JSON) 관리
# -----------------------------------------------------------------------------
def default_record():
    return {
        "data_source": DATA_SOURCE,
        "data_url": DATA_URL,
        "target": TARGET,
        "target_meaning": DEFAULT_MEANINGS[TARGET],
        "variable_meanings": DEFAULT_MEANINGS.copy(),
    }


def save_record(record):
    """내_기록 폴더를 만들고 JSON 파일에 기록한다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=4)


def load_record():
    """JSON 파일이 없으면 기본 기록을 생성하고, 있으면 불러온다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    defaults = default_record()

    if not RECORD_FILE.exists():
        save_record(defaults)
        return defaults

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)
        if not isinstance(saved, dict):
            raise ValueError("JSON의 최상위 구조가 객체가 아닙니다.")

        # 새 버전에서 변수가 추가되더라도 기존 사용자의 기록을 유지한다.
        merged = defaults.copy()
        merged.update(saved)
        saved_meanings = saved.get("variable_meanings", {})
        if not isinstance(saved_meanings, dict):
            saved_meanings = {}
        merged["variable_meanings"] = defaults["variable_meanings"].copy()
        merged["variable_meanings"].update(saved_meanings)
        return merged
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.warning(
            f"저장된 JSON 기록을 읽지 못해 기본 기록을 사용합니다. 오류: {error}"
        )
        save_record(defaults)
        return defaults


# -----------------------------------------------------------------------------
# 데이터 로딩 및 전처리
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv():
    """UTF-8 인코딩으로 원본 CSV를 읽는다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


def convert_target_value(value):
    """defects 값을 0(결함 없음), 1(결함 있음)으로 통일한다."""
    if pd.isna(value):
        return pd.NA

    if isinstance(value, bool):
        return int(value)

    # CSV에 숫자 0/1로 저장되어 있는 경우도 처리한다.
    if isinstance(value, (int, float)):
        if value == 0:
            return 0
        if value == 1:
            return 1

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "결함 있음"}:
        return 1
    if text in {"false", "0", "no", "n", "결함 없음"}:
        return 0
    return pd.NA


def prepare_dataframe(raw_df):
    """중복 행과 예측 대상 결측/비정상 행을 정리한다.

    입력 변수의 결측값은 여기서 전체 데이터 기준으로 채우지 않는다.
    모델 학습 시 train 데이터에만 맞춘 SimpleImputer로 처리해 데이터 누수를 막는다.
    """
    if TARGET not in raw_df.columns:
        raise ValueError(f"예측 대상 열인 '{TARGET}'이 CSV에 없습니다.")

    df = raw_df.copy()
    df[TARGET] = df[TARGET].apply(convert_target_value).astype("Int64")

    duplicate_count = int(df.duplicated().sum())
    after_dedup = df.drop_duplicates().copy()

    invalid_target_count = int(after_dedup[TARGET].isna().sum())
    cleaned = after_dedup.dropna(subset=[TARGET]).copy()
    cleaned[TARGET] = cleaned[TARGET].astype(int)

    feature_columns = [column for column in cleaned.columns if column != TARGET]
    return cleaned, feature_columns, duplicate_count, invalid_target_count


def get_feature_types(data, feature_columns):
    """숫자형과 범주형 입력 변수를 실제 데이터 형식에 따라 구분한다."""
    numeric_columns = []
    categorical_columns = []

    for column in feature_columns:
        series = data[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns


def make_preprocessor(X_train, model_name):
    """학습 데이터의 자료형에 맞춰 결측값 처리 및 인코딩 파이프라인을 만든다."""
    numeric_columns, categorical_columns = get_feature_types(
        X_train, list(X_train.columns)
    )
    transformers = []

    if numeric_columns:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_name == "로지스틱 회귀":
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=numeric_steps)
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("모델 학습에 사용할 수 있는 입력 변수가 없습니다.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_classifier(model_name):
    """클래스 불균형을 고려한 분류 모델을 생성한다."""
    if model_name == "로지스틱 회귀":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )

    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def calculate_numeric_bounds(X_train):
    """사용자 입력값이 학습 데이터 범위에서 벗어나는지 확인할 통계량."""
    bounds = {}
    numeric_columns, _ = get_feature_types(X_train, list(X_train.columns))

    for column in numeric_columns:
        values = pd.to_numeric(X_train[column], errors="coerce").dropna()
        if values.empty:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        bounds[column] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "lower_fence": q1 - 1.5 * iqr,
            "upper_fence": q3 + 1.5 * iqr,
        }
    return bounds


def train_and_evaluate(cleaned_df, feature_columns, model_name):
    """8:2 분할, 학습, 테스트 평가를 한 번에 수행한다."""
    X = cleaned_df[feature_columns].copy()
    y = cleaned_df[TARGET].astype(int).copy()

    if y.nunique() != 2:
        raise ValueError("결함 있음/없음 두 종류의 데이터가 모두 있어야 합니다.")
    if y.value_counts().min() < 2:
        raise ValueError("각 분류에 최소 2개 이상의 데이터가 있어야 분할할 수 있습니다.")

    # 모든 값이 결측인 입력 변수는 학습할 정보가 없으므로 제외한다.
    empty_columns = [column for column in X.columns if X[column].notna().sum() == 0]
    if empty_columns:
        X = X.drop(columns=empty_columns)
    if X.shape[1] == 0:
        raise ValueError("결측값이 아닌 데이터가 있는 입력 변수가 없습니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = make_preprocessor(X_train, model_name)
    classifier = make_classifier(model_name)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    return {
        "model": model,
        "model_name": model_name,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "y_test": y_test.tolist(),
        "predictions": predictions.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": list(X.columns),
        "dropped_empty_columns": empty_columns,
        "bounds": calculate_numeric_bounds(X_train),
        "class_counts": y.value_counts().sort_index().to_dict(),
    }


# -----------------------------------------------------------------------------
# 화면 구성 함수
# -----------------------------------------------------------------------------
def show_metric_card(title, value, help_text=None):
    st.metric(title, value, help=help_text)


def show_record_editor(record, all_columns):
    """사용자가 변수 뜻과 출처를 수정하고 JSON 파일로 저장한다."""
    with st.expander("📝 내 기록 편집 및 저장", expanded=False):
        st.write(
            "변수의 우리말 뜻이나 데이터 출처를 수정한 뒤 저장할 수 있습니다. "
            "저장한 내용은 앱을 다시 실행해도 JSON 파일에서 불러옵니다."
        )
        with st.form("record_edit_form"):
            edited_meanings = {}
            left, right = st.columns(2)
            for index, column in enumerate(all_columns):
                target_column = left if index % 2 == 0 else right
                with target_column:
                    default_meaning = record.get("variable_meanings", {}).get(
                        column, "직접 의미를 입력하세요."
                    )
                    edited_meanings[column] = st.text_input(
                        f"{column}의 우리말 뜻",
                        value=str(default_meaning),
                        key=f"meaning_edit_{column}",
                    )

            edited_source = st.text_input(
                "데이터 출처 이름", value=record.get("data_source", DATA_SOURCE)
            )
            edited_url = st.text_input(
                "데이터 출처 주소", value=record.get("data_url", DATA_URL)
            )
            submitted = st.form_submit_button("JSON 기록 저장", type="primary")

        if submitted:
            updated_record = {
                "data_source": edited_source,
                "data_url": edited_url,
                "target": TARGET,
                "target_meaning": edited_meanings.get(TARGET, DEFAULT_MEANINGS[TARGET]),
                "variable_meanings": edited_meanings,
            }
            try:
                save_record(updated_record)
                st.session_state["record"] = updated_record
                st.success(f"기록을 저장했습니다: {RECORD_FILE.as_posix()}")
            except OSError as error:
                st.error(f"기록 파일을 저장하지 못했습니다. 폴더 권한을 확인하세요. ({error})")


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

    show_record_editor(record, list(raw_df.columns))
    st.caption(
        f"데이터 출처: [{record.get('data_source', DATA_SOURCE)}]"
        f"({record.get('data_url', DATA_URL)})"
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

    if st.button("모델 학습 시작", type="primary", use_container_width=True):
        try:
            with st.spinner("데이터를 나누고 모델을 학습하는 중입니다..."):
                result = train_and_evaluate(cleaned_df, feature_columns, selected_model)
            st.session_state["model_bundle"] = result
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
        st.info(
            "학습이 끝났습니다. 왼쪽 메뉴의 '모델 성능 평가'에서 평가 지표와 혼동 행렬을, "
            "'사용자 코드 결함 예측'에서 직접 입력한 값의 예측 결과를 확인할 수 있습니다."
        )
    else:
        st.info("아직 학습된 모델이 없습니다. 모델을 선택한 뒤 '모델 학습 시작'을 누르세요.")


def page_evaluation():
    st.header("3. 모델 성능 평가")
    bundle = st.session_state.get("model_bundle")
    if not bundle:
        st.warning("먼저 '머신러닝 모델 제작' 화면에서 모델을 학습해 주세요.")
        return

    metrics = bundle["metrics"]
    st.write(f"평가 모델: **{bundle['model_name']}** · 테스트 데이터 {bundle['test_rows']:,}개")

    metric_columns = st.columns(4)
    metric_columns[0].metric("정확도 (Accuracy)", f"{metrics['accuracy'] * 100:.2f}%")
    metric_columns[1].metric("정밀도 (Precision)", f"{metrics['precision'] * 100:.2f}%")
    metric_columns[2].metric("재현율 (Recall)", f"{metrics['recall'] * 100:.2f}%")
    metric_columns[3].metric("F1-score", f"{metrics['f1'] * 100:.2f}%")

    st.subheader("혼동 행렬 (Confusion Matrix)")
    matrix = bundle["confusion_matrix"]
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
    fig.update_layout(
        title="실제 분류와 모델 예측 비교",
        xaxis_title="모델 예측 결과",
        yaxis_title="실제 정답",
    )
    st.plotly_chart(fig, use_container_width=True)

    tn, fp = matrix[0]
    fn, tp = matrix[1]
    detail_columns = st.columns(4)
    detail_columns[0].metric("TN · 정상으로 예측", f"{tn:,}개")
    detail_columns[1].metric("FP · 결함으로 잘못 예측", f"{fp:,}개")
    detail_columns[2].metric("FN · 결함을 놓침", f"{fn:,}개")
    detail_columns[3].metric("TP · 결함을 찾아냄", f"{tp:,}개")

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

    recall_rate = metrics["recall"] * 100
    st.info(
        f"실제 결함 모듈 중 모델이 찾아낸 비율(재현율)은 {recall_rate:.2f}%입니다. "
        f"테스트 데이터의 결함 모듈 {fn + tp:,}개 중 {tp:,}개를 찾아냈습니다."
    )

    st.subheader("결과 해석 시 주의할 점")
    st.markdown(
        "- 결함이 없는 데이터가 훨씬 많으면 모든 데이터를 '결함 없음'으로 예측해도 정확도가 높게 나올 수 있습니다. "
        "따라서 결함을 놓치지 않는 정도를 보는 재현율과 F1-score도 함께 확인해야 합니다.\n"
        "- 재현율을 높이면 정상 모듈을 결함으로 잘못 판단하는 경우(FP)가 늘어날 수 있습니다. "
        "실제 사용 목적에 따라 FP와 FN 중 어느 오류가 더 큰 문제인지 고려해야 합니다.\n"
        "- 이 모델은 JM1 데이터의 통계적 특성을 학습한 결과입니다. 다른 언어·프로젝트·개발 환경에서는 "
        "성능이 달라질 수 있으며, 실제 소프트웨어 전체에 그대로 적용할 수는 없습니다.\n"
        "- 클래스 불균형 보정(class_weight)을 적용했으므로, 예측 확률을 실제 서비스의 결함 발생률처럼 "
        "해석해서는 안 됩니다."
    )


def page_prediction(cleaned_df, record):
    st.header("4. 사용자 코드 결함 예측")
    bundle = st.session_state.get("model_bundle")
    if not bundle:
        st.warning("먼저 '머신러닝 모델 제작' 화면에서 모델을 학습해 주세요.")
        return

    st.write(
        "JM1 데이터셋과 같은 입력 변수에 코드 모듈의 통계값을 입력하면 "
        "학습된 모델이 결함 발생 여부와 결함 예측 확률을 계산합니다."
    )
    feature_columns = bundle["feature_columns"]
    all_missing_dropped = bundle.get("dropped_empty_columns", [])
    usable_data = cleaned_df[feature_columns]
    numeric_columns, categorical_columns = get_feature_types(usable_data, feature_columns)
    meanings = record.get("variable_meanings", {})

    input_values = {}
    with st.form("prediction_input_form"):
        st.subheader("코드 특성값 입력")
        left, right = st.columns(2)
        for index, column in enumerate(feature_columns):
            target_column = left if index % 2 == 0 else right
            meaning = meanings.get(column, "데이터셋 원본 변수")
            with target_column:
                if column in numeric_columns:
                    values = pd.to_numeric(usable_data[column], errors="coerce").dropna()
                    default_value = float(values.median()) if not values.empty else 0.0
                    is_integer = (
                        not values.empty
                        and pd.api.types.is_integer_dtype(usable_data[column])
                    )
                    step = 1.0 if is_integer else 0.1
                    input_values[column] = st.number_input(
                        f"{column} · {meaning}",
                        value=default_value,
                        step=step,
                        format="%.4f",
                        help=f"학습 데이터 범위: {values.min():.4f} ~ {values.max():.4f}"
                        if not values.empty else "학습 데이터에 숫자값이 없습니다.",
                    )
                else:
                    # 원본 자료형을 유지해야 bool/범주형 값이 학습 때와 같은 값으로 전달된다.
                    options = usable_data[column].dropna().unique().tolist()
                    options = sorted(options, key=lambda value: str(value))
                    if not options:
                        options = [""]
                    mode_values = usable_data[column].dropna().mode()
                    default_value = mode_values.iloc[0] if not mode_values.empty else options[0]
                    default_index = options.index(default_value) if default_value in options else 0
                    input_values[column] = st.selectbox(
                        f"{column} · {meaning}",
                        options=options,
                        index=default_index,
                    )

        submitted = st.form_submit_button("결함 예측하기", type="primary", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([input_values], columns=feature_columns)
            prediction = int(bundle["model"].predict(input_df)[0])
            probabilities = bundle["model"].predict_proba(input_df)[0]
            classifier_classes = bundle["model"].named_steps["classifier"].classes_.tolist()
            defect_probability = float(probabilities[classifier_classes.index(1)])

            st.divider()
            st.subheader("예측 결과")
            result_left, result_right = st.columns([1, 2])
            with result_left:
                if prediction == 1:
                    st.error("결함 있음으로 예측")
                else:
                    st.success("결함 없음으로 예측")
            with result_right:
                st.metric("결함 발생 예측 확률", f"{defect_probability * 100:.2f}%")
                probability_fig = go.Figure(
                    go.Bar(
                        x=["결함 발생 확률"],
                        y=[defect_probability * 100],
                        text=[f"{defect_probability * 100:.2f}%"],
                        textposition="outside",
                    )
                )
                probability_fig.update_layout(
                    yaxis_title="확률(%)",
                    yaxis=dict(range=[0, 110]),
                    xaxis_title="",
                    showlegend=False,
                    height=300,
                )
                st.plotly_chart(probability_fig, use_container_width=True)

            if prediction == 1:
                st.warning(
                    "주의: 모델이 이 모듈을 결함이 있을 가능성이 있는 것으로 분류했습니다. "
                    "코드 검토와 테스트를 추가로 진행하는 것이 좋습니다."
                )
            else:
                st.success(
                    "모델은 결함 없음으로 분류했습니다. 다만 결함이 없다는 보장은 아니므로 "
                    "일반적인 코드 리뷰와 테스트는 계속 필요합니다."
                )

            unusual_values = []
            outside_training_range = []
            for column, value in input_values.items():
                if column not in bundle["bounds"]:
                    continue
                bound = bundle["bounds"][column]
                numeric_value = float(value)
                if numeric_value < bound["min"] or numeric_value > bound["max"]:
                    outside_training_range.append(column)
                if numeric_value < bound["lower_fence"] or numeric_value > bound["upper_fence"]:
                    unusual_values.append(column)

            if unusual_values:
                st.warning(
                    "다음 입력값은 학습 데이터의 일반적인 범위(IQR 기준)를 크게 벗어납니다: "
                    + ", ".join(unusual_values)
                    + ". 이런 값에서는 예측의 신뢰도가 낮을 수 있습니다."
                )
            elif outside_training_range:
                st.info(
                    "다음 입력값이 학습 데이터의 최솟값~최댓값 범위 밖에 있습니다: "
                    + ", ".join(outside_training_range)
                    + ". 결과를 신중하게 해석하세요."
                )
            else:
                st.caption("입력한 숫자값은 학습 데이터의 최솟값~최댓값 범위 안에 있습니다.")

            if all_missing_dropped:
                st.caption(
                    "학습 데이터에 값이 전혀 없어 제외된 변수: "
                    + ", ".join(all_missing_dropped)
                )

            st.info(
                "중요한 한계: 이 프로그램은 코드의 통계적 특성을 바탕으로 결함 가능성을 분류합니다. "
                "실제 버그가 있는 코드 줄이나 위치를 찾아내는 정적 분석 도구는 아닙니다."
            )
        except Exception as error:
            st.error(
                "입력값을 예측 모델에 전달하는 중 문제가 발생했습니다. "
                f"입력 변수와 데이터 형식을 확인하세요. 상세 오류: {error}"
            )


# -----------------------------------------------------------------------------
# 앱 실행
# -----------------------------------------------------------------------------
def main():
    st.title("🧩 소프트웨어 결함 예측 프로그램")
    st.caption("NASA PROMISE JM1 데이터 기반 소프트웨어 모듈 결함 분류")

    if "record" not in st.session_state:
        st.session_state["record"] = load_record()
    record = st.session_state["record"]

    with st.sidebar:
        st.header("메뉴")
        selected_page = st.radio(
            "화면 선택",
            [
                "데이터 분석 및 전처리",
                "머신러닝 모델 제작",
                "모델 성능 평가",
                "사용자 코드 결함 예측",
            ],
        )
        st.divider()
        st.caption("데이터 출처")
        st.markdown("[NASA PROMISE JM1.csv](" + DATA_URL + ")")
        st.caption("예측 대상: defects")

    try:
        with st.spinner("JM1 데이터셋을 불러오는 중입니다..."):
            raw_df = load_csv()
        if raw_df.empty:
            st.error("CSV 파일에 데이터가 없습니다.")
            return
        cleaned_df, feature_columns, duplicate_count, invalid_target_count = prepare_dataframe(raw_df)
    except Exception as error:
        st.error(
            "데이터를 불러오거나 전처리하지 못했습니다. 인터넷 연결과 데이터 주소, "
            f"CSV 형식을 확인하세요. 상세 오류: {error}"
        )
        st.info("데이터 주소: " + DATA_URL)
        return

    if not feature_columns:
        st.error("예측 대상 defects를 제외한 입력 변수가 없습니다.")
        return

    if selected_page == "데이터 분석 및 전처리":
        page_data_analysis(
            raw_df,
            cleaned_df,
            feature_columns,
            duplicate_count,
            invalid_target_count,
            record,
        )
    elif selected_page == "머신러닝 모델 제작":
        page_model(cleaned_df, feature_columns)
    elif selected_page == "모델 성능 평가":
        page_evaluation()
    else:
        page_prediction(cleaned_df, record)

    st.divider()
    st.caption(
        "참고: 모델의 예측은 데이터에서 학습한 통계적 결과이며, 실제 결함 여부를 확정하지 않습니다."
    )


if __name__ == "__main__":
    main()
