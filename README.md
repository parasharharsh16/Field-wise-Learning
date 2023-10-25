# Field-wise Learning for Multi-field Categorical Data
## Requirements
The code has been tested with:
- Python 3.6.8
- PyTorch 1.1.0
- lmdb 0.96
- tqdm 4.32.1
- matplotlib 2.8.2


## Training and Evaluation
1. Download the dataset for Adult Income from https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

2. To train and evaluate the model(s), run following command :
	
python run_fwl.py  --dataset-path ./data/AdultIncome/adult.csv --ebd-dim 1.6 --log-ebd --lr 0.01 --wdcy 1e-6 --include-linear --reg-lr 1e-3 --reg-mean --reg-adagrad


## Reference 
This code is modified version of orginal code published by authors at https://github.com/lzb5600/Field-wise-Learning

The modified version of this code to use Adult Income dataset is also available at my repositiory here: https://github.com/parasharharsh16/Field-wise-Learning