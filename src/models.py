"""
    Models used in Fake News Detection
"""
import os
import shutil
from typing import Dict 
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

class BERT():
    def __init__(self, loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=tf.metrics.BinaryAccuracy(), compile_model=True):
        self.model = self.build_model()
        self.loss = loss
        self.metrics = metrics 
        self.optimizer = self.build_optimizer()

        if compile_model:
            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                                metrics=self.metrics)

    def build_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)

    def build_optimizer(self):
        epochs = 5
        #steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = 1 * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        init_lr = 3e-5
        return optimization.create_optimizer(init_lr=init_lr,
                                             num_train_steps=num_train_steps,
                                             num_warmup_steps=num_warmup_steps,
                                             optimizer_type='adamw')


    def fit(self, training_text, training_label, epochs=100):
        return self.model.fit(training_text,
                              training_label,
                              validation_split=0.25,
                              epochs=epochs)

    def evaluate_model(self, test_text, test_label):
        return self.model.evaluate(test_text, test_label)

    def get_config(self) -> Dict:
        pass 
