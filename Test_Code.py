import torch
from torch import nn
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import torchvision.transforms as transforms

# Load the model's state dictionary
model_state_dict = torch.load('mnist_lr.pt')

# Define the logistic regression model class again
class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)  # 28x28 input to 10 output classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input image to a 1D vector
        x = self.fc(x)  # Pass through the fully connected layer
        return x

# Instantiate the model
model = LogisticRegressionModel()

# Load the state dictionary into the model
model.load_state_dict(model_state_dict)

# Set the model to evaluation mode
model.eval()

# Define a transform for the image
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# GUI Functions
def load_image():
    global image, img_path, img_tk, img_label, preview_canvas
    
    # Open file dialog to select an image
    img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    
    if not img_path:
        return
    
    # Load the image
    image = Image.open(img_path)
    img_tk = ImageTk.PhotoImage(image)
    
    # Display the image in a canvas for better UI
    preview_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    img_label.config(text="Image Loaded", fg="green")
    
def predict_digit():
    if not img_path:
        messagebox.showerror("Error", "Please load an image first.")
        return
    
    # Load and preprocess the image
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)  # Add batch dimension
    output = model(img)  # Predict
    _, predicted = torch.max(output, 1)
    predicted_digit = predicted.item()
    
    prediction_label.config(text=f"Predicted Digit: {predicted_digit}", fg="blue")

def reset():
    preview_canvas.delete("all")  # Clear the image from canvas
    prediction_label.config(text="")
    img_label.config(text="")
    global img_path
    img_path = ""

# Create main application window
root = tk.Tk()
root.title("MNIST Digit Predictor")

# Set window layout and make it resizable
root.geometry("500x400")  # Set initial size of the window
root.resizable(True, True)  # Allow resizing in both directions

# Create widgets
preview_canvas = tk.Canvas(root, width=250, height=250, bg="white", highlightthickness=0)
preview_canvas.pack(pady=10)

img_label = tk.Label(root, text="", font=("Helvetica", 12))
img_label.pack(pady=5)

load_button = tk.Button(root, text="Load Image", command=load_image, bg="#007BFF", fg="white", font=("Helvetica", 12, "bold"))
load_button.pack(pady=5)

predict_button = tk.Button(root, text="Predict", command=predict_digit, bg="#28A745", fg="white", font=("Helvetica", 12, "bold"))
predict_button.pack(pady=5)

prediction_label = tk.Label(root, text="", font=("Helvetica", 12))
prediction_label.pack(pady=5)

reset_button = tk.Button(root, text="Reset", command=reset, bg="#FFC107", fg="black", font=("Helvetica", 12, "bold"))
reset_button.pack(pady=10)

# Run the main loop
root.mainloop()
