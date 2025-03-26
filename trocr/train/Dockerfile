FROM apache/airflow:latest
ENV PATH="/opt/conda/bin:$PATH"
RUN pip install -r /app/requirements.txt
ENTRYPOINT ["airflow", "celery", "worker"]