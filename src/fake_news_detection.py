"""
    Fake News detection using several ML models (BERT, LSTM, CNN and other)
"""
import argparse
import json
import pandas as pd

from models import BERT, FF, loss_function, metric_function, perform_grid_tuning, ensembling_models

import tensorflow as tf
#physical_devices = tf.config.list_physical_devices('GPU')

#tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    parser.add_argument('-grid_search_models', type=str, default='../my_hyperparameters.json',
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
        
        if data["model_1"]:
            models.append(BERT(data["model_1"]))
        if data["model_2"]:
            models.append(BERT(data["model_2"]))
        
        if data["model_3"]:
            models.append(BERT(data["model_3"]))
        
        if data["model_4"]:
            models.append(BERT(data["model_4"]))
        if data["model_5"]:
            models.append(BERT(data["model_5"]))
        if data["model_6"]:
            models.append(BERT(data["model_6"]))
        if data["model_7"]:
            models.append(BERT(data["model_7"]))
        if data["model_8"]:
            models.append(BERT(data["model_8"]))
        if data["model_9"]:
            models.append(BERT(data["model_9"]))
        if data["model_10"]:
            models.append(BERT(data["model_10"]))
        
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
    train_data = train_data.append(validation_data)
    models = initialize_models(command_line_args.models)
    for index, model in enumerate(models):
        model.fit(train_data["tweet"], train_data["label"],
                  (test_data["tweet"], test_data["label"]))
        model.model.save('model_' + str(index))
    #models[0].plot_loss()
    #models[0].plot_accuracy()
    #print(models[0].evaluate_model(test_data["tweet"], test_data["label"]))
    #models_trained = [model.model for model in models]
    #print(len(models_trained))
    #ensembling_model = ensembling_models(models_trained)
    #early_stopping = tf.keras.callbacks.EarlyStopping(
    #        monitor='val_accuracy', patience=5, restore_best_weights=True)
    #ensembling_model.evaluate(validation_data["tweet"], validation_data["label"])
    #ensembling_model.save('ensembling')
    #with open(command_line_args.grid_search) as models_file:
    #    data = json.load(models_file)
    #    perform_grid_tuning(data, train_data["tweet"], train_data["label"],
    #                        (validation_data["tweet"], validation_data["label"]))

label_value = {
    'fake': 0,
    'real': 1,
}

if __name__ =="__main__":
    args = command_line_arguments()
    fake_news_detection(args)
