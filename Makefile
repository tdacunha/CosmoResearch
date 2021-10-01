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

run_example_5:
	python example_5_generate.py
	python example_5_run.py

run_examples:
	python example_1_run.py
	python example_2_run.py
	python example_3_run.py
	python example_4_run.py
	python example_5_run.py

generate_examples:
	python example_1_generate.py
	python example_2_generate.py
	python example_3_generate.py
	python example_4_generate.py
	python example_5_generate.py

example_2_video:
	python example_2_video.py
	convert -quality 100 results/example_2/video/*.png results/example_2/training_video.gif

clean:
	@rm -rf results/*
