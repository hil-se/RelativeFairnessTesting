import tensorflow as tf
import numpy as np

class VGG_Pre:
    def __init__(self, start_size = 64, input_shape = (224, 224, 3)):
        base_model = tf.keras.models.Sequential()
        base_model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                   input_shape=input_shape))
        base_model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                                   input_shape=input_shape))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        base_model.add(
            tf.keras.layers.Conv2D(start_size*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        base_model.add(
            tf.keras.layers.Conv2D(start_size*4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        base_model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        base_model.add(
            tf.keras.layers.Conv2D(4096, kernel_size=(7, 7), strides=(1, 1), padding='valid',
                                   activation='relu'))
        base_model.add(tf.keras.layers.Dropout(0.5))
        base_model.add(
            tf.keras.layers.Conv2D(4096, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   activation='relu'))
        base_model.add(tf.keras.layers.Dropout(0.5))
        base_model.add(
            tf.keras.layers.Conv2D(2622, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   activation='relu'))

        base_model.add(tf.keras.layers.Flatten())
        base_model.add(tf.keras.layers.Activation('softmax'))
        base_model.load_weights('checkpoint/vgg_face_weights.h5')

        for layer in base_model.layers[:-4]:
            layer.trainable = False

        base_model_output = tf.keras.layers.Flatten()(base_model.layers[-4].output)
        base_model_output = tf.keras.layers.Dense(256, activation="relu")(base_model_output)
        # base_model_output = tf.keras.layers.Dropout(0.5)(base_model_output)
        base_model_output = tf.keras.layers.Dense(1, activation='sigmoid')(base_model_output)

        self.model = tf.keras.Model(inputs=base_model.input, outputs=base_model_output)
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], optimizer='adam')

        # base_model_output = tf.keras.layers.Dense(1)(base_model_output)
        #
        # self.model = tf.keras.Model(inputs=base_model.input, outputs=base_model_output)
        # self.model.compile(loss='huber_loss', metrics=['rmse'], optimizer='adam')


    def fit(self, X, y, sample_weight=None):
        # pre-trained weights of vgg-face model.
        # you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
        # related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/


        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=1, mode='auto',
                                      min_lr=5e-5)

        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint/attractiveness.hdf5'
                                       , monitor="val_loss", verbose=0
                                       , save_best_only=True, mode='auto'
                                       )
        history = self.model.fit(X, y, sample_weight=sample_weight, callbacks=[lr_reduce,checkpointer], validation_split = 0.2, batch_size = 10, epochs=10)
        print(history.history)

    def predict(self, X):
        pred = self.model.predict(X)
        pred = (pred.flatten()>0.5).astype(int).astype(float)
        return pred

    def decision_function(self, X):
        pred = self.model.predict(X)
        return pred

    def load_model(self, checkpoint_filepath):
        self.model = tf.keras.models.load_model(checkpoint_filepath)


