FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8887

ENV FLASK_APP=src/app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=8887"]