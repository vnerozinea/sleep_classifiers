import os
import shutil
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as keras
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return unpack, (model, training_config, weights)

    cls = tf.keras.models.Model
    cls.__reduce__ = __reduce__


class QuantConstraint(tf.keras.constraints.Constraint):
    """
    Used for 8-bit weight quantization in Keras.
    Quantization helper functions provided in Figure 5-12 of the FPGA User Guide
    """

    def __init__(self, name="", **kwargs):
        super(QuantConstraint, self).__init__(**kwargs)
        self.name = name

    def __call__(self, w):
        with tf.compat.v1.variable_scope(self.name + "_CONSTRAINTS") as scope:
            return QuantConstraint.lin_8b_quant(w)

    @staticmethod
    def lin_8b_quant(w, min_rng=-0.5, max_rng=0.5):
        """
        8-bit activation quantization in Keras using Lambda layer
        """
        if min_rng == 0.0 and max_rng == 2.0:
            min_clip = 0
            max_clip = 255
        else:
            min_clip = -128
            max_clip = 127
        wq = 256.0 * w / (max_rng - min_rng)
        wq = keras.round(wq)  # integer (quantization)
        wq = keras.clip(wq, min_clip, max_clip)  # fit into 256 linear quantization
        wq = wq / 256.0 * (max_rng - min_rng)  # back to quantized real number, not integer
        wclip = keras.clip(w, min_rng, max_rng)  # linear value w/ clipping
        return wclip + keras.stop_gradient(wq - wclip)

    @staticmethod
    def act_quant_8b(x, a_bin=16, min_rng=0.0, max_rng=2.0):
        """
        For use in Lambda layer
        """
        return QuantConstraint.lin_8b_quant(x, min_rng=min_rng, max_rng=max_rng)


class KerasClassifier:

    def __init__(self, quantize=False, num_classes=3, force_pickable=False, export_all=True):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        self.quantize = quantize
        self.num_classes = num_classes
        self.model = None
        self.split_counter = 0
        self.export_all = export_all
        self.export_dir = 'exports'

        if not os.path.exists(self.export_dir):
            os.mkdir(self.export_dir)

        if force_pickable:
            make_keras_picklable()

    def fit(self, x, y, epochs=10):
        num_features = x.shape[1]

        tf.keras.backend.clear_session()

        if self.quantize:
            input_layer = tf.keras.Input(shape=(num_features,))
            t = tf.keras.layers.Dense(units=15, kernel_constraint=QuantConstraint("layer1"))(input_layer)
            t = tf.keras.layers.ReLU()(t)
            t = tf.keras.layers.Dense(units=15, kernel_constraint=QuantConstraint("layer2"))(t)
            t = tf.keras.layers.ReLU()(t)
            t = tf.keras.layers.Dense(units=15, kernel_constraint=QuantConstraint("layer3"))(t)
            t = tf.keras.layers.ReLU()(t)
            output = tf.keras.layers.Dense(units=self.num_classes, kernel_constraint=QuantConstraint("final_layer"))(t)
            quant_output = tf.keras.layers.Lambda(QuantConstraint.act_quant_8b)(output)
            self.model = tf.keras.Model(inputs=input_layer, outputs=quant_output)
        else:
            input_layer = tf.keras.Input(shape=(num_features,))
            t = tf.keras.layers.Dense(units=15)(input_layer)
            t = tf.keras.layers.ReLU()(t)
            t = tf.keras.layers.Dense(units=15)(t)
            t = tf.keras.layers.ReLU()(t)
            t = tf.keras.layers.Dense(units=15)(t)
            t = tf.keras.layers.ReLU()(t)
            output = tf.keras.layers.Dense(units=self.num_classes)(t)
            self.model = tf.keras.Model(inputs=input_layer, outputs=output)

        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        # self.model.summary()

        self.model.fit(x, y, epochs=epochs)
        # print(self.model.layers[1].weights)

    def predict(self, x, batch_size=None, verbose=0):
        probs = self.model.predict(x, batch_size=batch_size, verbose=verbose)
        classes = [np.argmax(p) for p in probs]
        if self.export_all:
            output_dir = f'{self.export_dir}/split_{self.split_counter:04d}'
            os.mkdir(output_dir)
            self.export(f'{output_dir}/model.h5')
            np.savez(f'{output_dir}/features.npz', features=x)
            np.savez(f'{output_dir}/probs.npz', probs=probs)
            np.savez(f'{output_dir}/classes.npz', classes=classes)
            self.split_counter += 1
        return classes

    def predict_proba(self, x, batch_size=None, verbose=0):
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def export(self, file_path='model.h5'):
        tf.keras.backend.set_learning_phase(0)  # use inference model format
        self.model.save(file_path, save_format='h5')

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
