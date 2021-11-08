import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() #splits data into 50k training and 10k testing

train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

'''
for i in range(9):
    examples = np.where(train_labels == i)
    first_index = examples[0][0]
    plt.imshow(train_images[first_index])
    plt.xlabel(class_names[train_labels[first_index][0]])
    plt.figure()
'''

#model
model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same",  input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same" ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10))

model.summary()

#data augmentation
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
#aug_train = datagen.flow(train_images, train_labels, batch_size=32)

#visualise data augmentation
#picking a single image to transform
test_img = train_images[38]
img = image.img_to_array(test_img)  # convert image to numpy arry
img = img.reshape((1,) + img.shape)  # reshape image

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  #this loops runs forever until we break
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 3:  # show 3 images
        break

plt.show()

#compile model
model.compile(optimizer=tf.keras.optimizers.Adadelta(
    learning_rate=1.0, rho=0.95, epsilon=1e-07),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(datagen.flow(train_images, train_labels, batch_size=B_S),
                    validation_data=(test_images, test_labels), steps_per_epoch=len(train_images) // B_S,
                    epochs=E)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
