FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# 파이썬 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드
COPY . .

# 로그 디렉토리
RUN mkdir -p logs

# 비특권 사용자
RUN useradd -m agent && chown -R agent:agent /app
USER agent

CMD ["python", "main.py"]
