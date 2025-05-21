# Image-Classifier
This project is a basic Convolutional Neural Network (CNN) image classifier built using TensorFlow and Keras. It was originally intended as a beginner project to demonstrate the steps required to load image data, build a CNN model, train it, and make predictions.

📁 Project Structure

Image_Classifier.ipynb
CNN/
  └── data/
      ├── train/
      │   ├── class_1/
      │   └── class_2/
      └── test/
          ├── class_1/
          └── class_2/

📌 Features (Original Version)

Manual image data loading using os and cv2

Defined a basic CNN using Conv2D, MaxPooling2D, Flatten, and Dense

Model trained on local images from Google Drive

Model saved using .h5 format and reloaded

Single image prediction with preprocessing using OpenCV

✅ Technologies Used

Python 3

TensorFlow / Keras

OpenCV (cv2)

NumPy

Google Colab (with Google Drive mounted)

🚀 How to Run

Mount Google Drive in Colab:

from google.colab import drive
drive.mount('/content/drive')

Ensure the following structure in your Drive:

/My Drive/CNN/data/train
/My Drive/CNN/data/test

Load the data and train the model from notebook cells

Save the model with:

model.save('imageclassifier.h5')

Load and predict using:

model = tf.keras.models.load_model('imageclassifier.h5')

🧠 Limitations (Original Version)

Model overfits quickly (no data augmentation)

No validation during training

Lacks metrics like precision, recall, confusion matrix

Old .h5 saving format

No visualization of training performance
