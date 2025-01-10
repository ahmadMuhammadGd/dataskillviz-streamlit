FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
build-essential \
curl \
&& rm -rf /var/lib/apt/lists/*

EXPOSE 8501

COPY requirements.txt .
RUN pip install -r requirements.txt

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "./src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]