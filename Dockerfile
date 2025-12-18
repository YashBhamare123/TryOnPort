FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 

WORKDIR /app

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

RUN pip install fastapi uvicorn gradio requests cloudinary python-dotenv pillow

COPY --chown=user . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]