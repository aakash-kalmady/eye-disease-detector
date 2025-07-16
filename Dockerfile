FROM python:3.11 as base

WORKDIR /app

COPY . .

RUN  python -m pip install -r requirements.txt

EXPOSE 8080

CMD python ./app.py
