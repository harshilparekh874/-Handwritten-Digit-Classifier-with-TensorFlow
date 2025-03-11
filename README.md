# **Handwritten Digit Classifier with TensorFlow**

This project implements an end-to-end machine learning pipeline for classifying handwritten digits using the MNIST dataset and TensorFlow's Keras API. It covers robust data preprocessing, neural network design, training, evaluation, and visualization. The trained model is saved as `mnist_model.h5` for integration into interactive applications.

---

## **Overview**

* **Data Ingestion & Preprocessing:**
  * Loads the MNIST dataset (60,000 training and 10,000 test images) via TensorFlow’s Keras API.
  * Flattens each 28×28 grayscale image into a 784-dimensional vector and scales pixel values to the [0, 1] range.
  * Applies one-hot encoding to convert digit labels (0–9) into categorical vectors.

* **Neural Network Design & Training:**
  * Constructs a Sequential model with two hidden dense layers (128 neurons each, ReLU activation) and a softmax output layer for 10-class classification.
  * Compiles the model using the SGD optimizer and categorical cross-entropy loss, and trains for 3 epochs.
  * Saves the trained model as `mnist_model.h5` for later deployment.

* **Evaluation & Visualization:**
  * Evaluates the model on the test set, reporting accuracy and loss.
  * Visualizes predictions using a 5×5 grid of test images (correct predictions in green, misclassifications in red).
  * Plots the softmax probability distribution for selected test images to inspect model confidence.

---

## **Technical Details**

* **Deep Neural Network Engineering:**  
  Architected an MLP using TensorFlow’s Keras API with dense layers and ReLU activations, culminating in a softmax layer for probabilistic multi-class classification.

* **Data Normalization & Feature Scaling:**  
  Transforms 28×28 images into 784-dimensional vectors and scales pixel intensities to ensure numerical stability during training.

* **Optimization & Training:**  
  Utilizes stochastic gradient descent (SGD) for iterative weight updates over 3 epochs with real-time monitoring of training accuracy and loss.

* **Model Serialization:**  
  Persists the trained model in HDF5 format (`mnist_model.h5`) using `model.save()`, ensuring reproducibility and enabling future deployment.

---

## **Installation**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/handwritten-digit-classifier.git
   cd handwritten-digit-classifier
Create & Activate a Virtual Environment:
(Ensure you are using a supported Python version, e.g., Python 3.8 - 3.10)

Windows (CMD):
bash
Copy
python -m venv tf_env
tf_env\Scripts\activate
macOS/Linux:
bash
Copy
python3 -m venv tf_env
source tf_env/bin/activate
Install Dependencies:

bash
Copy
pip install tensorflow matplotlib numpy Pillow
Usage
Training & Saving the Model
Run the training script to build, evaluate, and save the model:

bash
Copy
python HandrwittenDigitAnalyzer.py
This script:

Loads and preprocesses the MNIST dataset.
Trains the neural network for 3 epochs.
Evaluates model performance on the test set.
Saves the trained model as mnist_model.h5.
Running the Interactive Drawing Application
After training, launch the drawing interface for real-time digit classification:

bash
Copy
python drawyourownnumber.py
Draw a digit on the canvas and click Predict to see the model’s prediction.

Future Enhancements
Integrate convolutional neural networks (CNNs) to further improve accuracy.
Implement data augmentation techniques to enrich the training dataset.
Deploy the model via a web or mobile interface.
Explore advanced hyperparameter tuning and regularization methods.
