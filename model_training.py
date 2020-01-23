# Generic/Built-in

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
CHECKPOINT_PATH = "./training_v1/cp.ckpt"
vocab_encoder = tfds.features.text.SubwordTextEncoder.load_from_file('vocab')
VOCAB_SIZE = vocab_encoder.vocab_size
EMBEDDING_SIZE = 64
BATCH_SIZE = 250
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)


train = prepare_tf_dataset('train')
test = prepare_tf_dataset('test')
train = train.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([-1], []))
test = test.padded_batch(batch_size=BATCH_SIZE, padded_shapes=([-1], []))
print('training data size {}'.format(BATCH_SIZE*len(list(train))))
print('Testing data size {}'.format(BATCH_SIZE*len(list(test))))


def create_functional_gru_model(vocab_size=VOCAB_SIZE, embed_size=EMBEDDING_SIZE):
    input_review = tf.keras.Input(shape=(None, ), name='review')
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_size)(input_review)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, dropout=0.2, recurrent_dropout=0.2))(embedding)
    x = tf.keras.layers.Dense(units=32, activation='relu')(x)
    out = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=input_review, outputs=out, name='sentiment_model')
    
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
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
    model = create_functional_gru_model(VOCAB_SIZE, EMBEDDING_SIZE)
    history = model.fit(    
                        train, 
                        epochs=5, 
                        validation_data=test,
                        callbacks=[cp_callback]
                        ) 
    eval_results = model.evaluate(test)
    print(eval_results)

    plot_graphs(history)
    tf.saved_model.save(model, EXPORT_PATH)
    

if __name__ == '__main__':
    model_training()