install:
	pip install --upgrade pip && pip install -r requirements.txt

run:
	python3 src/main.py

clean:
	rm -rf __pycache__ venv