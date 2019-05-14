install : install-conda environment

install-conda :
	@setup/install-conda.sh

environment :
	source ~/anaconda3/etc/profile.d/conda.sh
	conda env create -f environment.yml
	setup/install-pytorch.sh

