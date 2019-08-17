# Graph Embedding Similarity Search
This folder contains all code regarding our work on using LSH to efficiently find nodes with similar graph embeddings.
To run experiments use **train.py**. For an overview on what options exist, 
just type `python train.py --help`. Please keep it mind that depending on from where you run this script, you might need to run this script via `PYTHONPATH=PYTHONPATH:[path to this repos root level] python train.py --help`.  
An example experiment could be `python train.py --dataset=Cora --lsh`.  
To start a more elaborate grid search over all datasets, run `python train.py --grid-search`
