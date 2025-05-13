# Variables
ARCHIVE_NAME = implementation
ARCHIVES_DIR = archives

# Declare phony targets
.PHONY: pack run

# Run 15 second pipeline
run:
	@echo "Running 15 second pipeline..."
	@pipenv run ./run_pipeline_15s.sh

# Pack the current directory respecting Git rules (.gitignore and .gitkeep files)
pack:
	@mkdir -p $(ARCHIVES_DIR)
	@echo "Packing repository contents..."
	@git ls-files | zip -q $(ARCHIVES_DIR)/$(ARCHIVE_NAME).zip -@
	@echo "Archive created: $(ARCHIVES_DIR)/$(ARCHIVE_NAME).zip"
