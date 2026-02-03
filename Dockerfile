FROM python:3.9-slim

WORKDIR /home/app

RUN apt-get update && apt-get install -y python3-tk tk && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


COPY . .
