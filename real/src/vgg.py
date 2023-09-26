import tensorflow as tf
import numpy as np

class VGG:
    def __init__(self, start_size = 64, input_shape = (350, 350, 3)):
        self.model = tf.keras.models.Sequential()
        self.model.add(
            # First pad to 352*352
            tf.keras.layers.ZeroPadding2D(padding=1, input_shape=input_shape))
        self.model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size*2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 2, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size*4, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 4, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8 * 8, kernel_size=(11, 11), strides=(1, 1), padding='valid',
                                   activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(
            tf.keras.layers.Conv2D(start_size * 8 * 8, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], optimizer='adam')


    def fit(self, X, y, sample_weight=None, base="P1"):
        # pre-trained weights of vgg-face model.
        # you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
        # related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

        # self.model.load_weights('checkpoint/vgg_face_weights.h5')
        # for layer in self.model.layers[:-4]:
        #     layer.trainable = False

        lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=8, verbose=1, mode='min',
                                      min_lr=5e-5)

        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoint/attractiveness_'+base+'.hdf5'
                                       , monitor="val_loss", verbose=1
                                       , save_best_only=True, mode='min'
                                       )
        history = self.model.fit(X, y, sample_weight=sample_weight, callbacks=[lr_reduce,checkpointer], validation_split = 0.1, batch_size=128, epochs=20)
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


