
black:
	python3 -m black src

clean:
	rm -rf __pycache__

lint:
	python3 -m pylint src

.PHONY: black clean

