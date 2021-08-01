pylint: ## lint code with pylint
	pylint nnrl -d similarities

pylint-test: ## lint test files with pylint
	pylint --rcfile=tests/pylintrc tests -d similarities

similarities: ## check code duplication with pylint
	poetry run pylint nnrl -d all -e similarities

reorder-imports-staged:
	git diff --cached --name-only | xargs grep -rl --include "*.py" 'import' | xargs reorder-python-imports --separate-relative

mypy-staged:
	git diff --cached --name-only | xargs grep -rl --include "*.py" 'import' | xargs mypy --follow-imports silent

poetry-export:
	poetry export --dev -f requirements.txt -o requirements.txt

poetry-update:
	poetry update
	make poetry-export
	git add pyproject.toml poetry.lock requirements.txt
	git commit -s -m "chore(deps): make poetry update"

bump-patch:
	poetry version patch
	git add pyproject.toml
	git commit -s -m "chore: bump version patch"

bump-minor:
	poetry version minor
	git add pyproject.toml
	git commit -s -m "chore: bump version minor"

bump-major:
	poetry version major
	git add pyproject.toml
	git commit -s -m "chore: bump version major"

changelog:
	git describe --abbrev=0 | xargs poetry run auto-changelog --tag-prefix v --unreleased --stdout --starting-commit

black:
	blackd 2> /dev/null &
