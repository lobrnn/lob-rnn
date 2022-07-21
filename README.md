# LOB RNN
This repo contains code to reproduce results from "A generative model of a limit order book using recurrent neural networks".

## Data preparation 
### Simulation of Markov chain data
Markov chain data is simulated using the script [mc_data_simulation.py](https://github.com/lobrnn/lob-rnn/blob/main/src/mc_data_simulation.py).

### Nasdaq data
The Nasdaq data is not publicly available, see https://data.houseoffinance.se/ for more information. Once the CSV data files are available, the Nasdaq data can be parsed using the script [parse_nasdaq_data.py](https://github.com/lobrnn/lob-rnn/blob/main/src/parse_nasdaq_data.py). This script expects the data files to be named by the format "NASDAQ_order_book_view_SE0000148884_YYYY-MM-DD.csv" where YYYY is the year, MM the month and DD the date. 

## Training the model
The script [train_rnn.py](https://github.com/lobrnn/lob-rnn/blob/main/src/train_rnn.py) is used to train a model. The script saves checkpoints of the model after each episode as well as .txt file with model info. 

## Evaluation of the model

### Generation of data
A pretrained model can be used to generate data using the script [generate_data_rnn.py](https://github.com/lobrnn/lob-rnn/blob/main/src/generate_data_rnn.py). The script saves the generated data in a pickle file and information about the generation is saved in a .txt file. 

The generated data is evaluated and visualized in jupyter notebooks, [evaluation_mc.ipynb](https://github.com/lobrnn/lob-rnn/blob/main/notebooks/evaluation_mc.ipynb) for the Markov chain model data and [evaluation_nasdaq.ipynb](https://github.com/lobrnn/lob-rnn/blob/main/notebooks/evaluation_nasdaq.ipynb) for the Nasdaq data. To run this notebook with your own trained model, change the timestamps to match your model and make sure that the data files are in the expected folder. 

### TWAP simulation 
For the Markov chain model training data, one of the evaluations is to run a TWAP simulation using the original Markov chain model and the trained RNN model. This is done in the script [twap_simulation.py](https://github.com/lobrnn/lob-rnn/blob/main/src/twap_simulation.py). The script saves results in a pickle file and information about the simulation is saved in a .txt file. The results are visualized in the notebook [twap.ipynb](https://github.com/lobrnn/lob-rnn/blob/main/notebooks/twap.ipynb). To run this notebook with your own trained model, change the timestamps to match your model and make sure that the data files are in the expected folder.

### Mid price prediction
The script [prediction.py](https://github.com/lobrnn/lob-rnn/blob/main/src/prediction.py) is used to perform the mid price prediction using a trained RNN model as well as training and predicting using the DeepLOB model. The results are summarized in tables in the notebook [prediction.ipynb](https://github.com/lobrnn/lob-rnn/blob/main/notebooks/prediction.ipynb). To run this notebook with your own trained model, change the timestamps to match your model and make sure that the data files are in the expected folder.
