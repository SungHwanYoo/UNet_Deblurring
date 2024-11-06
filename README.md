<img width="911" alt="cat" src="https://github.com/user-attachments/assets/62f348c2-fbb4-48ec-a51e-894f1f3592a7">

# UNet based Deblurring

### Setting

The Kaggle Dogs vs. Cats dataset, originally designed for binary classification, was repurposed to create a robust image restoration framework.

By artificially introducing Gaussian blur to these high-quality pet images, I established a controlled environment for training and evaluating deblurring algorithms.

This strategic adaptation of the dataset is particularly compelling as it provides a diverse collection of natural images with rich textures, complex features, and varying lighting conditions 

---

### Setup

This code has been tested with Python 3.8.8, Torch 1.10.0

- Setup requirements

```
pip install -r requirements.txt
```

---

### Public Datasets

Dogs vs. Cats : ```https://www.kaggle.com/c/dogs-vs-cats```

---

### Start Training

```
python train.py
```
