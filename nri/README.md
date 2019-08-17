# Neural Relational Inference (NRI)
To train a NRI model on either the spring-dataset from the original paper or our weather data, use the **train.py** in this folder.  
You can either just get an overview on possible changable parameters, run `python train.py --help`. All manually supplied parameters will overwrite those, specified in the **config.json**. Alternatively, modify hyperparameters and other settings in there and run the script via `python train.py --config [path to your modified config.json]`.  
As the data sets are rather large, you have to first download them and drop them off in a new folder (per default **datasets**, located on this same level). Then just supply the path to the data file via `--dataset-path=[path to file]`.  

An example experiment (on the weather dataset that works on the smoothed time series and uses 4 latent interaction types) could be invoked with the following command: `python train.py --dataset-name=weather --n-edges=4 --dataset-path=[path-to-dataset-folder] --weather-data-suffix=exp_moving_avg`.
