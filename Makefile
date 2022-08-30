NBCONVERT=docker run --user root --rm -v $(pwd):/code -w /code jupyter/minimal-notebook python3 -m nbconvert  --to markdown --ClearOutputPreprocessor.enabled=True

generate_md:
	@echo "Generating md files from notebooks"
	docker-compose run --rm notebook -c \
		'find . -name "*.ipynb"|xargs -n 1 -Ifile python3 -m nbconvert --to markdown --ClearOutputPreprocessor.enabled=True file'