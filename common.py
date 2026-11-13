"""페이지 간 공유 기능: 데이터 로딩, 전처리, 모델 학습, 평가, JSON 기록 관리."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
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


def make_classifier(model_name, use_class_weight=True):
    """클래스 불균형을 고려한 분류 모델을 생성한다."""
    weight = "balanced" if use_class_weight else None
    
    if model_name == "로지스틱 회귀":
        return LogisticRegression(
            max_iter=2000,
            class_weight=weight,
            random_state=42,
        )

    return RandomForestClassifier(
        n_estimators=300,
        class_weight=weight,
        random_state=42,
        n_jobs=-1,
    )


def train_and_evaluate(cleaned_df, feature_columns, model_name, use_log=False, use_cap=False, use_smote=False, use_class_weight=True):
    """8:2 분할, 학습, 테스트 평가를 수행하며 고급 전처리 및 가중치 옵션을 적용한다."""
    X = cleaned_df[feature_columns].copy()
    y = cleaned_df[TARGET].astype(int).copy()

    if y.nunique() != 2:
        raise ValueError("결함 있음/없음 두 종류의 데이터가 모두 있어야 합니다.")
    if y.value_counts().min() < 2:
        raise ValueError("각 분류에 최소 2개 이상의 데이터가 있어야 분할할 수 있습니다.")

    empty_columns = [column for column in X.columns if X[column].notna().sum() == 0]
    if empty_columns:
        X = X.drop(columns=empty_columns)
    if X.shape[1] == 0:
        raise ValueError("결측값이 아닌 데이터가 있는 입력 변수가 없습니다.")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    numeric_columns, _ = get_feature_types(X_train, list(X_train.columns))
    original_bounds = calculate_numeric_bounds(X_train)

    # 1. 이상치 제어 (Capping)
    cap_bounds = {}
    if use_cap and numeric_columns:
        cap_bounds = X_train[numeric_columns].quantile(0.99).to_dict()
        for col in numeric_columns:
            X_train[col] = X_train[col].clip(upper=cap_bounds[col])
            X_test[col] = X_test[col].clip(upper=cap_bounds[col])

    # 2. 로그 변환 (Log Transform)
    if use_log and numeric_columns:
        import numpy as np
        X_train[numeric_columns] = np.log1p(X_train[numeric_columns].astype(float))
        X_test[numeric_columns] = np.log1p(X_test[numeric_columns].astype(float))

    preprocessor = make_preprocessor(X_train, model_name)
    classifier = make_classifier(model_name, use_class_weight)

    # 3. SMOTE 적용 (imbalanced-learn)
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            model = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ])
        except ImportError:
            raise ImportError("SMOTE 기능을 사용하려면 터미널에서 'pip install imbalanced-learn'을 실행하세요.")
    else:
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    classifier_classes = model.named_steps["classifier"].classes_.tolist()
    defect_probabilities = probabilities[:, classifier_classes.index(1)]
    predictions = (defect_probabilities >= 0.5).astype(int)

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
        "defect_probabilities": defect_probabilities.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": list(X.columns),
        "dropped_empty_columns": empty_columns,
        "bounds": original_bounds,
        "class_counts": y.value_counts().sort_index().to_dict(),
        "use_log": use_log,
        "use_cap": use_cap,
        "cap_bounds": cap_bounds,
    }


def evaluate_at_threshold(bundle, threshold):
    """저장된 테스트 확률에 임계값을 적용해 성능을 다시 계산한다."""
    y_true = bundle["y_test"]
    probabilities = bundle["defect_probabilities"]
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    return {
        "predictions": predictions,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


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


def load_app_data():
    """JM1 원본과 전처리 결과를 반환한다. 페이지마다 같은 캐시 데이터를 사용한다."""
    raw_df = load_csv()
    if raw_df.empty:
        raise ValueError("CSV 파일에 데이터가 없습니다.")
    cleaned_df, feature_columns, duplicate_count, invalid_target_count = prepare_dataframe(raw_df)
    if not feature_columns:
        raise ValueError("예측 대상 defects를 제외한 입력 변수가 없습니다.")
    return raw_df, cleaned_df, feature_columns, duplicate_count, invalid_target_count

def drop_high_correlation_features(df, feature_columns, threshold=0.9):
    """상관관계가 높은 중복 변수를 찾아 제거할 목록과 축소된 변수 목록을 반환한다."""
    numeric_columns, _ = get_feature_types(df, feature_columns)
    if not numeric_columns:
        return feature_columns, [], None

    # 부호와 상관없이 중복성을 확인하기 위해 절댓값 사용
    corr_matrix = df[numeric_columns].corr().abs()
    
    # 대각선 기준 우상단 행렬만 추출해 A-B 중복 시 하나만 제거되도록 처리
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
    reduced_features = [col for col in feature_columns if col not in to_drop]

    return reduced_features, to_drop, corr_matrix
