# 베이스 이미지 설정
FROM python:3.11

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 파일 복사 및 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY . /app/

# 포트 노출
EXPOSE 8000

# 명령 실행
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
