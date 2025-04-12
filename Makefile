# Variables
ARCHIVE_NAME = repository-archive
TIMESTAMP = $(shell date +%Y%m%d-%H%M%S)
ARCHIVES_DIR = archives

# Declare phony targets
.PHONY: pack

# Pack the current directory respecting Git rules (.gitignore and .gitkeep files)
pack:
	@mkdir -p $(ARCHIVES_DIR)
	@echo "Packing repository contents..."
	@git ls-files | tar -czf $(ARCHIVES_DIR)/$(ARCHIVE_NAME)-$(TIMESTAMP).tar.gz -T -
	@echo "Archive created: $(ARCHIVES_DIR)/$(ARCHIVE_NAME)-$(TIMESTAMP).tar.gz"
