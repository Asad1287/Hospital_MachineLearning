filename = "filename"
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

server:
	python main.py