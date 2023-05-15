# Relative Fairness Testing

#### Data (included in the [data/](https://github.com/hil-se/RelativeFairnessTesting/tree/main/real/data) folder)

 - [SCUT-FBP5500](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release).
   + [Selected_Ratings.csv](https://github.com/hil-se/RelativeFairnessTesting/blob/main/real/data/Selected_Ratings.csv) extracts P1, P2, P3, and Average ratings from the original data.

#### Pre-Trained weights

 - The VGG-16 model utilizes pre-trained weights on ImageNet data from [deepface_models](https://github.com/serengil/deepface_models).

#### Usage
0. Install dependencies:
```
pip install -r requirements.txt
```
1. Create a folder checkpoint:
```
mkdir checkpoint
```
2. Download the pre-trained weights of VGG-16 model [vgg_face_weights.h5](https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5) and put it under _checkpoint/_
3. Navigate to the source code:
```
cd src
```
4. Generate results in [results/](https://github.com/hil-se/RelativeFairnessTesting/tree/main/real/results)
```
python main.py
```

