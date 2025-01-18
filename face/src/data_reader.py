import pandas as pd
import tensorflow as tf
image_path = "/local/datasets/idai720/images/"

def load_scut(rating_cols = ["P1", "P2", "P3", "Average"]):

    def retrievePixels(path):
        img = tf.keras.utils.load_img(image_path + path, target_size=(224, 224), grayscale=False)
        x = tf.keras.utils.img_to_array(img)
        return x

    data0 = pd.read_csv('../data/Ratings.csv')
    data = pd.DataFrame({"Filename": data0["Filename"]})
    for col in rating_cols:
        data[col] = data0[col]

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

