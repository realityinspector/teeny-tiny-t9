.PHONY: install race rigor clean all

install:
	pip install numpy scipy numpy-stl

race:
	python main.py

rigor:
	python rigor.py

all: race rigor

clean:
	rm -f *.stl rigor_results.json
	rm -rf __pycache__
