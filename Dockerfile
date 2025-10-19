FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt .

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . .

CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers=1", "--threads=1", "app:app"]
