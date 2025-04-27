FROM python:3.8-slim
WORKDIR /app
COPY bot/requirements_bot.txt /app
COPY bot/bot.py /app
RUN pip install -U pip && \
    pip install -r /app/requirements_bot.txt
CMD ["python", "bot.py"]