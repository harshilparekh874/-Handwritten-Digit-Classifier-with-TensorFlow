import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model('mnist_model.h5')

class DrawApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a Digit")
        self.canvas_width = 200
        self.canvas_height = 200
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Create a PIL image to store the drawing (black background)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        # Add buttons for prediction and clearing the canvas
        self.predict_button = tk.Button(self, text="Predict", command=self.predict)
        self.predict_button.pack(pady=5)
        self.clear_button = tk.Button(self, text="Clear", command=self.clear)
        self.clear_button.pack(pady=5)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=0)

    def predict(self):
        # Resize the image to 28x28 pixels
        resized_image = self.image.resize((28, 28), Image.LANCZOS)
        # Convert the image to a NumPy array and normalize pixel values to [0, 1]
        image_array = np.array(resized_image).astype('float32') / 255.0
        # Flatten the array to match the model input shape (1, 784)
        image_array = image_array.reshape(1, 784)
        
        # Visualize the preprocessed input
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        plt.title('Preprocessed Input')
        plt.show()
        
        # Make a prediction using the loaded model
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        print("Predicted digit:", predicted_digit)

if __name__ == '__main__':
    app = DrawApp()
    app.mainloop()
