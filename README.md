# SDXL Image Generator

SDXL 모델을 사용하여 텍스트 프롬프트로부터 이미지를 생성하는 웹 애플리케이션입니다.

## 기술 스택

- FastAPI
- PyTorch
- Stable Diffusion XL
- HTML/JavaScript

## 설치 방법

1. Python 가상환경 생성 및 활성화:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
.\venv\Scripts\activate  # Windows
```

2. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

## 실행 방법

1. 서버 실행:

```bash
python main.py
```

2. 웹 브라우저에서 다음 주소로 접속:

```
http://localhost:8000
```

## 사용 방법

1. 웹 페이지에서 텍스트 프롬프트 입력
2. "Generate Image" 버튼 클릭
3. 생성된 이미지 확인

## 주의사항

- SDXL 모델은 처음 실행 시 약 6GB의 모델 파일을 다운로드합니다.
- GPU가 있는 경우 자동으로 GPU를 사용하며, 없는 경우 CPU에서 실행됩니다.
- 이미지 생성에는 GPU 사용을 권장합니다.
