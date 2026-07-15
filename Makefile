.PHONY: help backup backup-list clean-backups

PYTHON ?= python3
BACKUP_DIR ?= backups
PROJECT_NAME := $(shell $(PYTHON) -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["name"])')
PROJECT_VERSION := $(shell $(PYTHON) -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
BACKUP_FILE := $(BACKUP_DIR)/$(PROJECT_NAME)-$(PROJECT_VERSION)-$(TIMESTAMP).tar.gz

BACKUP_EXCLUDES := \
	--exclude='./.git' \
	--exclude='./.agents' \
	--exclude='./.codex' \
	--exclude='./.idea' \
	--exclude='./.venv' \
	--exclude='./__pycache__' \
	--exclude='./.pytest_cache' \
	--exclude='./build' \
	--exclude='./dist' \
	--exclude='./docs/_build' \
	--exclude='./$(BACKUP_DIR)' \
	--exclude='*.egg-info' \
	--exclude='*.pyc' \
	--exclude='*.pyo' \
	--exclude='*/__pycache__' \
	--exclude='*.pyc' \
	--exclude='*.aux' \
	--exclude='*.log' \
	--exclude='*.fls' \
	--exclude='*.fdb_latexmk' \
	--exclude='*.synctex.gz'

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make backup        Create a timestamped source backup archive.' \
		'  make backup-list   List generated backup archives.' \
		'  make clean-backups Remove generated backup archives.'

backup:
	@mkdir -p "$(BACKUP_DIR)"
	tar $(BACKUP_EXCLUDES) -czf "$(BACKUP_FILE)" .
	@printf 'Created backup: %s\n' "$(BACKUP_FILE)"

backup-list:
	@find "$(BACKUP_DIR)" -maxdepth 1 -type f -name '*.tar.gz' -print 2>/dev/null | sort || true

clean-backups:
	rm -f "$(BACKUP_DIR)"/*.tar.gz
