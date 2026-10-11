.PHONY: verify validate train notebooks test

verify:
	python scripts/verify_repository.py

validate:
	python external_validation/run_external_validation.py

train:
	python training/train_cv.py --epochs 80 --batch-size 64

notebooks:
	jupyter lab notebooks/

test:
	pytest -q
