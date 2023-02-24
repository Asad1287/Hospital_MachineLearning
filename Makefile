inference_dataset = "/workspaces/Hospital_BusinessCaseStudy/LengthOfStay_Prod.csv"
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv test_hello.py

format:
	black *.py

lint:
	pylint --disable=R,C hello.py

hello:
	echo "Hello, world!" $(filename)

all: install lint test

train:
	python train_script.py

inference:
	python test_script.py $(inference_dataset)

server:
	python main.py