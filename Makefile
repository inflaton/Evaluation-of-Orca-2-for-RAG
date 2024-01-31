.PHONY: start
start:
	python app.py

test:
	python qa_chain_test.py

chat:
	python qa_chain_test.py chat

ingest:
	python ingest.py

.PHONY: format
format:
	black .

install:
	pip install -r requirements.txt
	cd ragas_extended && pip install -e .

install-mac:
	pip install -r requirements-mac.txt
	cd ragas_extended && pip install -e .
