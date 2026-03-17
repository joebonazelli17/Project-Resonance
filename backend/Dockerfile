FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir essentia==2.1b6.dev1389

# Download model files at build time
# CLAP checkpoint (~600MB), BERT tokenizer vocab, RoBERTa model (~500MB), Demucs (~80MB)
RUN mkdir -p /models/clap /models/bert-base-uncased /models/roberta-base /models/bart-base \
             /root/.cache/torch/hub/checkpoints && \
    curl -L -o /models/clap/music_audioset_epoch_15_esc_90.14.pt \
    "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt" && \
    curl -L -o /models/bert-base-uncased/vocab.txt \
    "https://raw.githubusercontent.com/microsoft/SDNet/master/bert_vocab_files/bert-base-uncased-vocab.txt" && \
    curl -L -o /models/roberta-base/config.json \
    "https://huggingface.co/roberta-base/resolve/main/config.json" && \
    curl -L -o /models/roberta-base/pytorch_model.bin \
    "https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin" && \
    curl -L -o /models/roberta-base/merges.txt \
    "https://huggingface.co/roberta-base/resolve/main/merges.txt" && \
    curl -L -o /models/roberta-base/vocab.json \
    "https://huggingface.co/roberta-base/resolve/main/vocab.json" && \
    curl -L -o /models/bart-base/vocab.json \
    "https://huggingface.co/facebook/bart-base/resolve/main/vocab.json" && \
    curl -L -o /models/bart-base/merges.txt \
    "https://huggingface.co/facebook/bart-base/resolve/main/merges.txt" && \
    curl -L -o /root/.cache/torch/hub/checkpoints/955717e8-8726e21a.th \
    "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"

# Bundle tokenizer config files
COPY models/bert-base-uncased/tokenizer_config.json /models/bert-base-uncased/tokenizer_config.json
COPY models/bert-base-uncased/config.json /models/bert-base-uncased/config.json
COPY models/roberta-base/tokenizer_config.json /models/roberta-base/tokenizer_config.json
COPY models/bart-base/tokenizer_config.json /models/bart-base/tokenizer_config.json

COPY . .

ENV CLAP_CKPT=/models/clap/music_audioset_epoch_15_esc_90.14.pt
ENV CLAP_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]