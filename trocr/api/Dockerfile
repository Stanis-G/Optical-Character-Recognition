FROM python:3.8-slim
WORKDIR /app
ARG MODEL_NAME=trocr_model
COPY trocr/api/requirements_api.txt /app
RUN pip install -U pip && \
    pip install -r /app/requirements_api.txt
COPY trocr/train/${MODEL_NAME} /app/${MODEL_NAME}
COPY trocr/api/api.py /app
COPY trocr/utils/utils_inf.py /app
CMD ["python", "api.py"]