import pandas as pd
import tensorflow as tf


def load_scut():

    def retrievePixels(path):
        # img = tf.keras.utils.load_img("../data/images/"+path, grayscale=False)
        img = tf.keras.utils.load_img("../data/images/" + path, target_size=(224, 224), grayscale=False)
        x = tf.keras.utils.img_to_array(img)
        return x

    data = pd.read_csv('../data/Selected_Ratings.csv')

    # discretize ratings (>3):
    rating_cols = ["Average", "P1", "P2", "P3"]
    for col in rating_cols:
        data[col] = data[col].apply(lambda x: 1 if x > 3 else 0)

    # extract sensitive attributes (Male=1, Female=0, Asian=1, Caucasian=0)
    sex = []
    race = []
    for file in data["Filename"]:
        if file[0]=='A':
            race.append(1)
        else:
            race.append(0)
        if file[1]=='M':
            sex.append(1)
        else:
            sex.append(0)
    protected = ['sex', 'race']
    data['sex'] = sex
    data['race'] = race
    data['pixels'] = data['Filename'].apply(retrievePixels)
    return data, protected

