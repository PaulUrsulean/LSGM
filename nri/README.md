# Neural Relational Inference (NRI)
To train a NRI model on either the spring-dataset from the original paper or our weather data, use the **train.py** in this folder.  
You can either just get an overview on possible changable parameters, run `python train.py --help`. All manually supplied parameters will overwrite those, specified in the **config.json**. Alternatively, modify hyperparameters and other settings in there and run the script via `python train.py --config [path to your modified config.json]`.  
As the data sets are rather large, you have to first download them and drop them off in a new folder (per default **datasets**, located on this same level). Then just supply the path to the data file via `--dataset-path=[path to file]`.  
Please keep it mind that depending on from where you run this script, you might need to run it via `PYTHONPATH=PYTHONPATH:[path to this repo's root level] python train.py`.

## Spring Dataset
The necessary files to run spring experiments can be generated as explained in the author's orgiginal repository under https://github.com/ethanfetaya/NRI. Just drop off all generated .npy files under **datsets/springs/**.

## Weather Data
Unfortunately the original dataset is not hosted anymore by the original website. To just directly use our processed data you can download a pickle file from [S3 (we will be hosting it for the next few days)](https://testbucket-ag97.s3.eu-central-1.amazonaws.com/100000_5_100_1_0_rawexp_moving_avg.pickle). Drop it off under **datsets/weather/** and invoke experiments with `python train.py --dataset-name=weather--dataset-path=[path-to-dataset-folder] --weather-data-suffix=exp_moving_avg`.
