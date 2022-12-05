FROM python:3.10.1


WORKDIR /fastapi
COPY requirements.txt /fastapi
RUN pip install -r requirements.txt
COPY ./app /fastapi/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
