from keras.models import load_model
import pickle
import re


'''
def load_tokenizer():
	from pathlib import Path
	dir_path = Path(__file__).parent
	tokenizer_path = ""
	full_path = dir_path / tokenizer_path
    train_path = "train.csv"
    full_train_path = dir_path / train_path
    test_path = "test.csv"
    full_test_path = dir_path / test_path
	with open(str(full_path),'rb') as handle:
		tokenizer = pickle.load(handle) 
	return tokenizer
'''

def load_dictionary():
    from pathlib import Path
	dir_path = Path(__file__).parent
	model_path = "word_index_dic.pickle"
    full_path = dir_path / model_path
    with open(str(full_path),'rb') as handle:
        loaded_dictionary = pickle.load(handle)
	return loaded_dictionary


def load_keras():
	from pathlib import Path
	dir_path = Path(__file__).parent
	model_path = "my_model.h5"
	full_path = dir_path / model_path
	model = load_model(str(full_path))
	return model