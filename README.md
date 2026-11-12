# 소프트웨어 결함 예측 프로그램

## 이 프로그램은

도농 고등학교에서 진행하는 데이터로 문제 해결하기 수행평가를 위해 만든거임ㅁㅁ
ChatGPT를 이용한 바이브 코딩으로 제작함

## 실행 방법

프로젝트 루트 폴더에서 아래 명령을 실행합니다.
(스트림릿 사이트에서 자동으로 실행 가능)

```bash
pip install -r requirements.txt
streamlit run main.py
```

## 폴더 구조

```text
software-defect-predictor/
├── main.py                 # 홈 화면 및 페이지 라우터
├── common.py               # 공통 데이터/전처리/모델/기록 기능
├── requirements.txt
├── pages/
│   ├── model/
│   │   ├── processing.py   # 데이터 분석 및 전처리
│   │   ├── make.py         # 머신러닝 모델 제작
│   │   └── rating.py       # 모델 성능 평가
│   └── use/
│       └── predict.py      # 사용자 코드 결함 예측
└── 내_기록/                # 첫 실행 시 JSON 파일 자동 생성
```

`common.py`에는 페이지들이 공유하는 데이터 로딩, 전처리, 모델 학습, 임계값 평가, JSON 기록 관리 코드가 있습니다.
모델과 임계값은 `st.session_state`를 통해 페이지 사이에 공유됩니다.
