FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "scripts/evaluate.py"]
