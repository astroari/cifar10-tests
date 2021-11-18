#!pip install -q pydub

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, MaxPool2D, Flatten, Dropout, Dense
from IPython.display import Audio
from matplotlib import pyplot as plt
from tqdm import tqdm

import pandas as pd

import librosa
import librosa.display

dataset =  tfds.load("gtzan_music_speech")

train = dataset["train"]
idata = iter(train)  #iterates through audio files
ex = next(idata)
audio = ex.get("audio")
label = ex.get("label")
Audio(audio, rate=22050)

index_class = {0: "music", 1: "speech"}
class_index = {"music" : 0, "speech": 1}

def plot_wave(audio):
    plt.plot(audio)
    plt.xlabel("samples")
    plt.ylabel("amplitude")
    plt.show()
plot_wave(audio)

#fourier transform
def stft(audio, frame_length = 2048, frame_step=512, fft_length=256):
    return tf.signal.stft(
    tf.cast(audio, tf.float32),
    frame_length= frame_length,
    frame_step= frame_step,
    fft_length= fft_length
    )
audio_stft = stft(audio)
audio_spec = tf.abs(audio_stft)

#spectrogram
def spec(spec):
    plt.figure(figsize=(12,14))
    plt.imshow(tf.transpose(spec), cmap= "viridis")
    plt.colorbar()
    plt.show()
spec(audio_spec[:200])

audio_spec_log = tf.math.log(audio_spec)
spec(audio_spec_log[:200])

def spectrogram(audio):
    audio_stft = stft(audio)
    audio_spec = tf.abs(audio_stft)
    return tf.math.log(tf.transpose(audio_spec))
    
#applying preprocessing to the data in chunks
sr =22050
chunk = 5 
def preprocess(ex):
    audio = ex.get("audio")
    label = ex.get("label")
    x_batch, y_batch = None,None
    for i in range (0, 6):
        start = i * chunk * sr
        end = (i + 1) * chunk * sr
        audio_chunk = audio[start: end]
        audio_spec = spectrogram(audio_chunk)
        audio_spec = tf.expand_dims(audio_spec, axis=0)
        current_label = tf.expand_dims(label, axis=0)
        x_batch = audio_spec if x_batch is None else tf.concat([x_batch, audio_spec], axis=0)
        y_batch = current_label if y_batch is None else tf.concat([y_batch, current_label], axis=0)
        return x_batch, y_batch
        
x_train, y_train = None, None
for ex in tqdm(iter(train)):
    x_batch, y_batch = preprocess(ex)
    x_train = x_batch if x_train is None else tf.concat([x_train, x_batch], axis=0)
    y_train= y_batch if y_train is None else tf.concat([y_train, y_batch], axis=0)
    
indices = tf.random.shuffle(list(range(0, 128)))
x_train = tf.gather(x_train, indices)
y_train = tf.gather(y_train, indices)
n_val = 300
x_valid=x_train[:n_val, ...]
y_valid=y_train[:n_val, ...]

plt.figure(figsize=(12,12))
st=0
for i in range(0,6):
    x,y = x_train[st +i], y_train[st+i]
    plt.subplot(3,2,i+1)
    plt.imshow(x, cmap="viridis")
    plt.title(index_class[y.numpy()])
    plt.colorbar()
plt.show()

input_ = Input(shape=(129, 212))
x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_)
for i in range(0, 4):
    num_filters = 2**(5 + i)
    x = Conv2D(num_filters, 3)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D(2)(x)
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(1, activation="sigmoid")(x)
model = tf.keras.models.Model(input_,x)

model.compile(
loss="binary_crossentropy",
optimizer= tf.keras.optimizers.Adam(learning_rate=3e-6),
metrics=["accuracy"]
)
model.summary()

history = model.fit(
    x_train, y_train,
    validation_data= (x_valid, y_valid),
    batch_size=5,
    epochs=10,
    verbose=True)
    
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
