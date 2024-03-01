INSTALLER=miniconda.sh
CONDA_BIN=./miniconda/bin/conda
RUN_SCRIPT=./run.sh

.PHONY:  clean notebook

OS := $(shell uname)
ifeq ($(OS), Linux)
 	URL := "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
else ifeq ($(OS), Darwin)
 	URL := "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
endif

$(INSTALLER):
	curl $(URL) -L -o $(INSTALLER) 
	chmod +x $(INSTALLER)
	chmod +x $(RUN_SCRIPT)

install: 
	./$(INSTALLER) -b -p ./miniconda
	$(CONDA_BIN) env create --file environment.yml

	
clean:
	rm -rf $(INSTALLER) miniconda
