# Generic/Built-in
import os

# Other Libs
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Own modules
from utils import prepare_tf_dataset


__author__ = "Ming Gao"
__credits__ = ["Ming Gao"]
__license__ = ""
__version__ = "0.1.0"
__maintainer__ = "Ming Gao"
__email__ = "ming_gao@outlook.com"
__status__ = "Dev"

'''
Create and train a bidirectional LSTM model
'''

EXPORT_PATH = "./saved_model/v1"
CHECKPOINT_DIR = "./training_v1"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")
vocab_encoder = tfds.features.text.SubwordTextEncoder.load_from_file('vocab')
VOCAB_SIZE = vocab_encoder.vocab_size
EMBEDDING_SIZE = 64
BATCH_SIZE = 250
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PREFIX,
                                                 save_weights_only=True,
                                                 verbose=1)


train = prepare_tf_dataset('train')
test = prepare_tf_dataset('test')
train = train.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([-1], []))
test = test.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([-1], []))
print('training data size {}'.format(BATCH_SIZE*len(list(train))))
print('Testing data size {}'.format(BATCH_SIZE*len(list(test))))


def create_functional_model(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE, dropout_rate=0.2):
    input_review = tf.keras.Input(shape=(None, ), name='review')
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)(input_review)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(embedding)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, activation='tanh'))(embedding)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Dense(units=32, activation='relu')(x)
    out = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_review, outputs=out, name='sentiment_model')
    
    model.compile(loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics = ['accuracy', tf.keras.metrics.AUC()])
    
    print(model.summary())
    
    return model


def plot_graphs(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.show()
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.legend(['auc', 'val_auc'])
    plt.show()


def model_training():
    model = create_functional_model(VOCAB_SIZE, EMBEDDING_SIZE)
    history = model.fit(    
                        train, 
                        epochs=5, 
                        validation_data=test,
                        callbacks=[cp_callback]
                        ) 
    plot_graphs(history)


def save_model():
    model_to_save = create_functional_model(VOCAB_SIZE, EMBEDDING_SIZE, dropout_rate=0)
    model_to_save.load_weights(tf.train.latest_checkpoint(CHECKPOINT_DIR))
    eval_results = model_to_save.evaluate(test)
    print(eval_results)

    tf.saved_model.save(model_to_save, EXPORT_PATH)


if __name__ == '__main__':
    model_training()
    save_model()