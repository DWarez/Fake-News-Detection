"""
    Fake News detection using several ML models (BERT, LSTM, CNN and other)
"""
import argparse
import json
import pandas as pd

from models import BERT, FF, loss_function, metric_function, perform_grid_tuning

import tensorflow as tf

# To use GPU
sess = tf.compat.v1.Session(
    config=tf.compat.v1.ConfigProto(log_device_placement=True))

def command_line_arguments():
    """
        Parse Command Line arguments provided when calling fake_news_detection program and 
        return an object with all parameters 
        Command Line arguments provided are the following:
        -Name of Training dataset path
        -Name of Test dataset path 
        -Path to JSON file with models and their parameters to detect Fake news 
        -Path to JSON file with Grid search choices 
    """
    parser = argparse.ArgumentParser(description='Fake News detection using some ML models')
    parser.add_argument('-training', metavar='TRAIN_PATH', type=str,
                        default='../data/train_set_covid.csv',
                        help='Path of Training dataset (Default data/train_set_covid.csv)',
                        dest='training_path')
    parser.add_argument('-validation', metavar='VAL_PATH', type=str,
                        default='../data/validation_set_covid.csv',
                        help='Path of Validation dataset (Default data/validation_set_covid.csv)',
                        dest='validation_path')
    parser.add_argument('-test', default='../data/test_set_covid.csv',
                        dest='test_path', help='Path of Test dataset (Default data/test_set_covid.csv)')
    parser.add_argument('-models', type=str, default='../default_models.json',
                        help='Path to JSON file with models and their parameters to detect Fake News',
                        dest='models')
    parser.add_argument('-grid_search_models', type=str, default='../hyperparameters.json',
                        help='Path to JSON file with Grid Search choices for model tuning',
                        dest='grid_search')
    return parser.parse_args()


def initialize_models(models_config):
    """
        Initialize models and returns list of model objects based on 
        JSON file with models and their parameters
        It pass to each model the JSON object of each model and the model 
        recognize and process the JSON object with parameters of the model  
    """
    models = []
    with open(models_config) as models_file:
        data = json.load(models_file)
        #if data["NNLM"]:
        #    models.append(FF(data["NNLM"]))
        if data["BERT"]:
            models.append(BERT(data["BERT"]))
    return models


def fake_news_detection(command_line_args):
    """
        Perform Fake News detection using         
    """
    train_data = pd.read_csv(command_line_args.training_path)[["tweet", "label"]]
    validation_data = pd.read_csv(command_line_args.validation_path)[["tweet", "label"]]
    test_data = pd.read_csv(command_line_args.test_path)[["tweet", "label"]]
    train_data["label"] = [label_value[label] for label in train_data["label"]]
    validation_data["label"] = [label_value[label] for label in validation_data["label"]]
    test_data["label"] = [label_value[label] for label in test_data["label"]]
    #models = initialize_models(command_line_args.models)
    #models[0].fit(train_data["tweet"], train_data["label"],
    #              (validation_data["tweet"], validation_data["label"]))
    #models[0].plot_loss()
    #models[0].plot_accuracy()
    #print(models[0].evaluate_model(test_data["tweet"], test_data["label"]))
    with open(command_line_args.grid_search) as models_file:
        data = json.load(models_file)
        perform_grid_tuning(data, train_data["tweet"], train_data["label"],
                        (validation_data["tweet"], validation_data["label"]))

label_value = {
    'fake': 0,
    'real': 1,
}

if __name__ =="__main__":
    args = command_line_arguments()
    fake_news_detection(args)
