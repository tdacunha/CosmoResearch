# Makefile with some useful targets:

run_example_1:
	python example_1_generate.py
	python example_1_run.py

run_example_2:
	python example_2_generate.py
	python example_2_run.py

run_example_3:
	python example_3_generate.py
	python example_3_run.py

run_example_4:
	python example_4_generate.py
	python example_4_run.py

run_examples:
	python example_1_run.py
	python example_2_run.py
	python example_3_run.py
	python example_4_run.py

generate_examples:
	python example_1_generate.py
	python example_2_generate.py
	python example_3_generate.py
	python example_4_generate.py

clean:
	@rm -rf results/*
