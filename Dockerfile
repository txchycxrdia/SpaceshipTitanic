FROM python:3.11-slim

WORKDIR /app

COPY ./spaceship_titanic-1.0-py3-none-any.whl /app

RUN pip install --no-cache-dir spaceship_titanic-1.0-py3-none-any.whl

COPY . /app

CMD ["python", "model.py", "run"]