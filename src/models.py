"""
    Models used in Fake News Detection
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text 
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

TFHUB_HANDLE_ENCODER = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
TFHUB_HANDLE_PREPROCESS = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

class BERT():
    def __init__(self, loss=tf.keras.losses.BinaryCrossentropy(),
                 metrics=tf.metrics.BinaryAccuracy(), epochs=50,
                 hub_bert_encoder=TFHUB_HANDLE_ENCODER,
                 hub_bert_preprocess=TFHUB_HANDLE_PREPROCESS, prob_dropout=0.1):
        self.model = self.build_model(prob_dropout, hub_bert_preprocess, hub_bert_encoder)
        self.loss = loss
        self.metrics = metrics
        self.optimizer = self.build_optimizer()
        self.epochs = epochs

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

    def build_model(self, prob_dropout, bert_preprocess, bert_encoder):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(bert_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(bert_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(prob_dropout)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)

    def build_optimizer(self):
        epochs = 50
        #steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = 10 * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        init_lr = 3e-5
        return optimization.create_optimizer(init_lr=init_lr,
                                             num_train_steps=num_train_steps,
                                             num_warmup_steps=num_warmup_steps,
                                             optimizer_type='adamw')


    def fit(self, training_text, training_label):
        return self.model.fit(training_text,
                              training_label,
                              validation_split=0.25,
                              epochs=self.epochs)

    def evaluate_model(self, test_text, test_label):
        return self.model.evaluate(test_text, test_label)

loss_function = {
    'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
}

metric_function = {
    'binary_accuracy': tf.metrics.BinaryAccuracy(),
}
