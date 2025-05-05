install:
	pip install -r requirements.txt
train:
	python scripts/train_teacher.py
distill:
	python scripts/distill.py
