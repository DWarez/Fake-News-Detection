"""
    Fake News detection using several ML models (BERT, LSTM, CNN and other)
"""
import argparse
import json
import pandas as pd
from models import ALBERT, BERT, loss_function, metric_function


def command_line_arguments():
    """
        Parse Command Line arguments provided when calling fake_news_detection program and 
        return an object with all parameters 
        Command Line arguments provided are the following:
        -Name of Training dataset path
        -Name of Test dataset path 
        -Path to JSON file with models and their parameters to detect Fake news 
    """
    parser = argparse.ArgumentParser(description='Fake News detection using some ML models')
    parser.add_argument('-training', metavar='TRAIN_PATH', type=str,
                        default='../data/train_set.csv',
                        help='Path of Training dataset (Default data/train_set.csv)',
                        dest='training_path')
    parser.add_argument('-test', default='../data/test_set.csv',
                        dest='test_path', help='Path of Test dataset (Default data/test_set.csv)')
    parser.add_argument('-models', type=str, default='../default_models.json',
                        help='Path to JSON file with models and their parameters to detect Fake News',
                        dest='models')
    #parser.add_argument('-parameters')
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
        if data["BERT"]:
            models.append(BERT(data["BERT"]))
        # if data["ALBERT"]:
        #     models.append(ALBERT(data["ALBERT"]))

    return models


def fake_news_detection(command_line_args):
    """
        Perform Fake News detection using         
    """
    train_data = pd.read_csv(command_line_args.training_path)[["text", "label"]]
    test_data = pd.read_csv(command_line_args.test_path)[["text", "label"]]

    models = initialize_models(command_line_args.models)
    models[0].fit(train_data["text"], train_data["label"])
    models[0].plot_loss()
    models[0].plot_accuracy()
    print(models[0].evaluate_model(test_data["text"], test_data["label"]))


if __name__ =="__main__":
    args = command_line_arguments()
    fake_news_detection(args)
