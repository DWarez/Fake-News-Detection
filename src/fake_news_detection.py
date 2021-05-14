"""
    Fake News detection using several ML models (BERT, LSTM, CNN and other)
"""
import argparse
import json
import pandas as pd
from models import BERT, loss_function, metric_function

def command_line_arguments():

    parser = argparse.ArgumentParser(description='Fake News detection using some ML models')
    parser.add_argument('-training', metavar='TRAIN_PATH', type=str,
                        default='../data/train_set.csv',
                        help='Name of Training dataset (Default data/train_set.csv)',
                        dest='training_path')
    parser.add_argument('-test', default='../data/test_set.csv',
                        dest='test_path', help='Name of Test dataset (Default data/test_set.csv)')
    parser.add_argument('-models', type=str, default='../default_models.json',
                        help='JSON file with models and their parameters to detect Fake News',
                        dest='models')
    #parser.add_argument('-parameters')
    return parser.parse_args()

def initialize_models(models_config):
    models = []
    with open(models_config) as models_file:
        data = json.load(models_file)
        if data["BERT"]:
            models.append(BERT(data["BERT"]))
    return models

#def initialize_bert(bert_config):
#    params = {}
#    params["loss"] = loss_function[bert_config["loss"]]
#    params["metrics"] = metric_function[bert_config["metrics"]]
#    params["epochs"] = bert_config["epochs"]
#    params["epochs_tuning"] = bert_config["epochs_tuning"]
#    params["bert_encoder"] = bert_config["encoder_hub"]
#    params["bert_preprocess"] = bert_config["preprocessing_hub"]
#    params[""] = bert_config["prob_dropout"]
#    return BERT(loss, metrics, epochs, bert_encoder, bert_preprocessing, prob_dropout)

def fake_news_detection(command_line_args):
    train_data = pd.read_csv(command_line_args.training_path)[["text", "label"]]
    test_data = pd.read_csv(command_line_args.test_path)[["text", "label"]][:500]

    models = initialize_models(command_line_args.models)
    models[0].fit(train_data["text"], train_data["label"])
    print(models[0].evaluate_model(test_data["text"], test_data["label"]))

if __name__ =="__main__":
    args = command_line_arguments()
    fake_news_detection(args)
