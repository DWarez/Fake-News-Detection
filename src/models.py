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
    def __init__(self, params):
        self.model = self.build_model(params["prob_dropout"], params["preprocessing_hub"], 
                                      params["encoder_hub"])
        self.loss = params["loss"]
        self.metrics = params["metrics"]
        self.optimizer = self.build_optimizer(params["epochs_tuning"], params["initial_lr"],
                                              params["optimizer_type"])
        self.epochs = params["epochs"]
        self.validation_split = params["validation_split"]
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
        self.history = []

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

    def build_optimizer(self, epochs_tuning, initial_learning_rate, optimizer):
        epochs = epochs_tuning
        #steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
        num_train_steps = 10 * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        init_lr = initial_learning_rate
        return optimization.create_optimizer(init_lr=init_lr,
                                             num_train_steps=num_train_steps,
                                             num_warmup_steps=num_warmup_steps,
                                             optimizer_type=optimizer)


    def fit(self, training_text, training_label):
        self.history = self.model.fit(training_text,
                              training_label,
                              validation_split=self.validation_split,
                              epochs=self.epochs)
        return self.history

    def evaluate_model(self, test_text, test_label):
        return self.model.evaluate(test_text, test_label)

    def plot_loss(self):
        history_dict = self.history.history

        loss = history_dict['loss']
        val_loss = history_dict['val_loss']

        epochs = range(1, self.epochs + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'r', label='Training loss')  
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        history_dict = self.history.history 
        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']
        epochs = range(1, self.epochs + 1)
        
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show() 
        
loss_function = {
    'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
}

metric_function = {
    'binary_accuracy': tf.metrics.BinaryAccuracy(),
}
