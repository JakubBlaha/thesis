# Variables
ARCHIVE_NAME = implementation
ARCHIVES_DIR = archives

# Declare phony targets
.PHONY: pack

# Pack the current directory respecting Git rules (.gitignore and .gitkeep files)
pack:
	@mkdir -p $(ARCHIVES_DIR)
	@echo "Packing repository contents..."
	@git ls-files | zip -q $(ARCHIVES_DIR)/$(ARCHIVE_NAME).zip -@
	@echo "Archive created: $(ARCHIVES_DIR)/$(ARCHIVE_NAME).zip"
