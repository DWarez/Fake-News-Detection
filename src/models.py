"""
    Models used in Fake News Detection
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text 
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt

TFHUB_HANDLE_BERT_ENCODER = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1"
TFHUB_HANDLE_BERT_PREPROCESS = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

TFHUB_HANDLE_ALBERT_ENCODER = "https://tfhub.dev/tensorflow/albert_en_base/3"
TFHUB_HANDLE_ALBERT_PREPROCESS = "http://tfhub.dev/tensorflow/albert_en_preprocess/3"


class BERT():
    """
        Model using BERT preprocessing and BERT encoding, with dropout and a dense layer for classification.
    """
    def __init__(self, params):
        self.model = self.build_model(params["prob_dropout"], params["BERT_preprocessing_hub"], 
                                      params["BERT_encoder_hub"])
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
        """Method for creating the model.

        Args:
            prob_dropout (float): Probability of Dropout
            bert_preprocess (str): Handler for BERT preprocessing model on TFHub
            bert_encoder (str): Handler for BERT encoding model on TFHub

        Returns:
            tf.keras.Model: Architecture of our BERT-based model
        """
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(bert_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(bert_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(prob_dropout)(net)
        net = tf.keras.layers.Dense(1, activation="tanh", name='classifier')(net)
        return tf.keras.Model(text_input, net)


    def build_optimizer(self, epochs_tuning, initial_learning_rate, optimizer):
        """Method for building the optimizer used for the BERT-based model

        Args:
            epochs_tuning (int): Number of epochs of training
            initial_learning_rate (float): Starting learning rate
            optimizer (str): Optimizer type used

        Returns:
            tf.keras.optimizer: Optimizer of the BERT-based model
        """
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
        """Method for fitting the model

        Args:
            training_text (): Patterns for training
            training_label (): Labels of training patterns

        Returns:
            tf.keras.callbacks.History: History of the training
        """
        self.history = self.model.fit(training_text,
                              training_label,
                              validation_split=self.validation_split,
                              epochs=self.epochs)
        return self.history


    def evaluate_model(self, test_text, test_label):
        """Evaluation of the model

        Args:
            test_text (): Patterns for testing
            test_label (): Labels of testing patterns

        Returns:
            float: Test loss
        """
        return self.model.evaluate(test_text, test_label)


    def plot_loss(self):
        """Method to plot the loss
        """
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
        """Method to plot the accuracy
        """
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


class ALBERT():
    """
        Model using ALBERT preprocessing and ALBERT encoding, with dropout and a dense layer for classification.
    """
    def __init__(self, params):
        self.model = self.build_model(params["prob_dropout"], params["ALBERT_preprocessing_hub"], 
                                      params["ALBERT_encoder_hub"])
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


    def build_model(self, prob_dropout, albert_preprocess, albert_encoder):
        """Method for creating the model.

        Args:
            prob_dropout (float): Probability of Dropout
            bert_preprocess (str): Handler for ALBERT preprocessing model on TFHub
            bert_encoder (str): Handler for ALBERT encoding model on TFHub

        Returns:
            tf.keras.Model: Architecture of our ALBERT-based model
        """
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(albert_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(albert_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(prob_dropout)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        return tf.keras.Model(text_input, net)


    def build_optimizer(self, epochs_tuning, initial_learning_rate, optimizer):
        """Method for building the optimizer used for the ALBERT-based model

        Args:
            epochs_tuning (int): Number of epochs of training
            initial_learning_rate (float): Starting learning rate
            optimizer (str): Optimizer type used

        Returns:
            tf.keras.optimizer: Optimizer of the ALBERT-based model
        """
        epochs = epochs_tuning
        num_train_steps = 10 * epochs
        num_warmup_steps = int(0.1*num_train_steps)

        init_lr = initial_learning_rate
        return optimization.create_optimizer(init_lr=init_lr,
                                             num_train_steps=num_train_steps,
                                             num_warmup_steps=num_warmup_steps,
                                             optimizer_type=optimizer)


    def fit(self, training_text, training_label):
        """Method for fitting the model

        Args:
            training_text (): Patterns for training
            training_label (): Labels of training patterns

        Returns:
            tf.keras.callbacks.History: History of the training
        """
        self.history = self.model.fit(training_text,
                              training_label,
                              validation_split=self.validation_split,
                              epochs=self.epochs)
        return self.history


    def evaluate_model(self, test_text, test_label):
        """Evaluation of the model

        Args:
            test_text (): Patterns for testing
            test_label (): Labels of testing patterns

        Returns:
            float: Test loss
        """
        return self.model.evaluate(test_text, test_label)


    def plot_loss(self):
        """Method to plot the loss
        """
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
        """Method to plot the accuracy
        """
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


# Loss Dictionary        
loss_function = {
    'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
}

# Metric Dictionary
metric_function = {
    'binary_accuracy': tf.metrics.BinaryAccuracy(),
}
