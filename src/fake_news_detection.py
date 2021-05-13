"""
    Fake News detection using several ML models (BERT, LSTM, CNN and other)
"""
#import tensorflow as tf 
import argparse 
import pandas as pd 
import json 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from models import BERT 

def command_line_arguments():

    parser = argparse.ArgumentParser(description='Fake News detection using some ML models')
    parser.add_argument('-training', metavar='TRAIN_PATH', type=str, default='../data/train_set.csv',
                        help='Name of Training dataset (Default data/train_set.csv)', dest='training_path')
    parser.add_argument('-test', default='../data/test_set.csv',
                        dest='test_path', help='Name of Test dataset (Default data/test_set.csv)')
    parser.add_argument('-models', type=str, default='default_models.json', 
                        help='JSON file with models and their parameters to use to detect Fake News', dest='models')
    #parser.add_argument('-parameters')
    return parser.parse_args()
    
def initialize_models(models_config):
    return BERT() 

def fake_news_detection(command_line_args):
    train_data = pd.read_csv(command_line_args.training_path)[["text", "label"]][:100]
    #test_data = pd.read_csv(command_line_args.test_path)[["text", "label"]][:500]

    models = initialize_models(command_line_args.models)
    models.fit(train_data["text"], train_data["label"], 20)
    #print(models.evaluate_model(test_data["text"], test_data["label"]))

if __name__== "__main__":
    args = command_line_arguments()
    fake_news_detection(args)
