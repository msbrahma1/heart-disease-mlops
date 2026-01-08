FROM public.ecr.aws/docker/library/python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask gunicorn requests numpy

COPY src/ ./src/
COPY reproduce_model.py .
COPY heart_disease_best_model.pkl .
COPY model_metadata.json .
COPY app.py .

EXPOSE 5000

HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]