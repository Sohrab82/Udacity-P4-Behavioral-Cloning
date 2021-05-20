import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import cv2
import ntpath
import sklearn
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, Cropping2D, SeparableConv2D
from keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import glob


def add_data(data_dir):
    # reads the csv files in data_dir and read the
    # file names and steering angles for each file
    with open(data_dir + '/driving_log.csv') as f:
        lines = f.readlines()
    stearing_angles = []
    file_names = []
    for line in lines[1:]:
        line = line.split(',')
        file_name = data_dir + '/IMG/' + ntpath.basename(line[0].strip())
        file_names.append(file_name)

        stearing_angle = float(line[3])
        stearing_angles.append(stearing_angle)

    return file_names, stearing_angles


# reading file names and steering angles for very data collection
stearing_angles = []
file_names = []
f, s = add_data('./data')
stearing_angles.extend(s)
file_names.extend(f)
f, s = add_data('./mydata')
stearing_angles.extend(s)
file_names.extend(f)
f, s = add_data('./mydata2')
stearing_angles.extend(s)
file_names.extend(f)
f, s = add_data('./mydata3')
stearing_angles.extend(s)
file_names.extend(f)
f, s = add_data('./mydata4')
stearing_angles.extend(s)
file_names.extend(f)

# apply moving average filter to steering angles
N = 3
moving_aves = []
cumsum = np.cumsum(stearing_angles)
for i in range(N, len(cumsum) - N):
    moving_ave = (cumsum[i] - cumsum[i - N]) / N
    moving_aves.append(moving_ave)

for i in range(len(file_names) - len(moving_aves)):
    moving_aves.append(moving_aves[-1])

# collect file names and angles into "samples"
samples = []
for i in range(len(file_names)):
    samples.append([file_names[i], moving_aves[i], 0])
    # add negative steering wheel angle to balance left & right turns
    # 0 & 1 are used in the generator function
    # 0 means original data
    # 1 means the image needs to be flipped left-right
    samples.append([file_names[i], -moving_aves[i], 1])


def generator(samples, batch_size=32):
    # generator function to be used for model.fit
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # batch_sample[0]: filename
                # batch_sample[0]: steering angle
                # batch_sample[0]: flip_lr the image if 1
                file_name = batch_sample[0]
                stearing_angle = batch_sample[1]

                image = cv2.imread(file_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                flip_st = batch_sample[2]
                if flip_st == 1:
                    image = np.fliplr(image)
                images.append(image)
                angles.append(stearing_angle)

            X_train = np.array(images, dtype=np.float16)
            y_train = np.array(angles)

            yield (X_train, y_train)


# split data into training & validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# load inception model from keras
freeze_flag = True  # `True` to freeze layers, `False` for full training
weights_flag = 'imagenet'  # 'imagenet' or None
preprocess_flag = True  # Should be true for ImageNet pre-trained typically
model = InceptionV3(weights=weights_flag, include_top=False,
                    input_shape=(85, 320, 3))
if freeze_flag == True:
    for l in model.layers:
        l.trainable = False

model.summary()


# Attaching new input layer with Lambda layer to the pretrained model
ch, row, col = 3, 160, 320  # Trimmed image format
input_layer = Input((row, col, ch))
x = input_layer
x = Lambda(lambda x: x / 255,
           input_shape=(row, col, ch),
           output_shape=(row, col, ch))(x)
x = Cropping2D(cropping=((55, 20), (0, 0)), input_shape=(row, col, ch))(x)
x = model(x)
x = SeparableConv2D(kernel_size=(1, 4), filters=32, activation='relu')(x)
# x = Conv2D(kernel_size=(1, 4), filters=64, activation='relu')(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dense(1)(x)

# create new_model and compile it
new_model = Model(inputs=input_layer, outputs=x)
adam = Adam(learning_rate=1e-3)
new_model.compile(loss='mse', optimizer=adam)
new_model.summary()

# fit the model to the data
new_model.fit(train_generator,
              steps_per_epoch=np.ceil(len(train_samples) / batch_size),
              validation_data=validation_generator,
              validation_steps=np.ceil(len(validation_samples) / batch_size),
              epochs=5, verbose=1)

# evaluate the model for the validation set
new_model.evaluate(validation_generator, steps=np.ceil(
    len(validation_samples) / batch_size))

# save the model
new_model.save('model.h5')


# predict the steering angle for some random images
for file_name in np.random.choice(glob.glob('./data/IMG/center*.jpg'), 10):
    image = cv2.imread(file_name)
    image = tf.expand_dims(image, 0)
    p_center = new_model.predict(image)
    print(p_center)
