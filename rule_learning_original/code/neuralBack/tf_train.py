import tensorflow as tf 
import pandas as pd  

class AndLayer(tf.keras.layers.Layer):
    def __init__(self, number_aux_pred):
        super(AndLayer, self).__init__()
        self.number_aux_pred = number_aux_pred

    def build(self, input_shape):
        w_init = tf.random_normal_initializer(mean=0.5, stddev=0.01)
        self.weights = tf.Variable(name="and_weights",   initial_value=w_init(shape=(input_shape[-1], self.number_aux_pred),dtype='float32'),trainable=True)


    def call(self, inputs):
        self.interpretable_rul_weights = tf.nn.softmax(self.weights, axis=0)
        w_times_x= tf.matmul(inputs, self.interpretable_rul_weights)
        biased = w_times_x - 0.5
        activated_aux_predicates =  2 * tf.nn.relu(biased)
        # activated_aux_predicates =  nn.functional.relu(w_times_x)
        return activated_aux_predicates


def train_single():
    data = pd.read_csv('rule_learning_original/code/DFORL/buzz/data/buzz/data4.csv')
    target = data.pop('label')
    x = data.values
    data = tf.data.Dataset.from_tensor_slices((x, target.values))
    data = data.shuffle(buffer_size=10000).batch(102400)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    loss = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),loss=loss,metrics=['binary_accuracy'])
    model.fit(data, epochs=10000)
    
train_single()