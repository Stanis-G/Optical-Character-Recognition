FROM python:3.8-slim
WORKDIR /app
ARG MODEL_NAME=trocr_model
COPY requirements_api.txt /app
RUN pip install -U pip && \
    pip install -r /app/requirements_api.txt
COPY ${MODEL_NAME} /app/${MODEL_NAME}
COPY trocr_api.py /app
COPY utils_inf.py /app
CMD ["python", "trocr_api.py"]