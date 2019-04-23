virtualenv -p python3.6 venv
source venv/bin/activate
pip install -r requirements.txt
ipython kernel install --user --name=autoencoder_mnist