import gradio as gr
import joblib  # scikit-learn
import tensorflow as tf  # tensorflow
import torch  # pytorch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# Load .pkl model
model1 = joblib.load("models/model.pkl")

# Load .h5 model
model2 = tf.keras.models.load_model("models/model.h5")

# Define the PyTorch model architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load .pt model and set it to evaluation mode
model3 = SimpleNN()
model3.load_state_dict(torch.load("models/model.pt"))
model3.eval()

# Generate bar chart for class probabilities
def generate_bar_chart(probabilities, classes):
    fig, ax = plt.subplots()
    ax.bar(classes, probabilities, color=['#ff9999','#66b3ff','#99ff99'])
    ax.set_xlabel("Classes")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probabilities")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# Generate pie chart for class probabilities
def generate_pie_chart(probabilities, classes):
    fig, ax = plt.subplots()
    ax.pie(probabilities, labels=classes, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    ax.axis('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# Prediction functions for each model
def predict_model1(sepal_length, sepal_width, petal_length, petal_width):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    probabilities = model1.predict_proba([features])[0]
    iris_classes = ["Setosa", "Versicolor", "Virginica"]
    predicted_class = iris_classes[np.argmax(probabilities)]
    bar_chart = generate_bar_chart(probabilities, iris_classes)
    pie_chart = generate_pie_chart(probabilities, iris_classes)
    return predicted_class, bar_chart, pie_chart

def predict_model2(sepal_length, sepal_width, petal_length, petal_width):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    features_array = np.array(features).reshape(1, -1)
    probabilities = model2.predict(features_array)[0]
    iris_classes = ["Setosa", "Versicolor", "Virginica"]
    predicted_class = iris_classes[np.argmax(probabilities)]
    bar_chart = generate_bar_chart(probabilities, iris_classes)
    pie_chart = generate_pie_chart(probabilities, iris_classes)
    return predicted_class, bar_chart, pie_chart

def predict_model3(sepal_length, sepal_width, petal_length, petal_width):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    features_tensor = torch.tensor(features, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        output = model3(features_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
    iris_classes = ["Setosa", "Versicolor", "Virginica"]
    predicted_class = iris_classes[np.argmax(probabilities)]
    bar_chart = generate_bar_chart(probabilities, iris_classes)
    pie_chart = generate_pie_chart(probabilities, iris_classes)
    return predicted_class, bar_chart, pie_chart

# Enhanced Gradio app layout with columns
with gr.Blocks() as demo:
    gr.Markdown("## 🌸 Iris Flower Classification")
    gr.Markdown("### Adjust the flower characteristics below to see real-time predictions and probability distributions for each model.")

    # Input Sliders
    with gr.Row():
        sepal_length = gr.Slider(4.0, 8.0, step=0.1, label="Sepal Length (cm)", value=5.1)
        sepal_width = gr.Slider(2.0, 4.5, step=0.1, label="Sepal Width (cm)", value=3.5)
        petal_length = gr.Slider(1.0, 7.0, step=0.1, label="Petal Length (cm)", value=1.4)
        petal_width = gr.Slider(0.1, 2.5, step=0.1, label="Petal Width (cm)", value=0.2)

    # Prediction Columns for each model
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model 1: Random Forest (Scikit-Learn)")
            output1_text = gr.Textbox(label="Predicted Class")
            output1_bar = gr.Image(label="Class Probabilities (Bar Chart)")
            output1_pie = gr.Image(label="Class Probabilities (Pie Chart)")

        with gr.Column():
            gr.Markdown("### Model 2: Neural Network (Keras)")
            output2_text = gr.Textbox(label="Predicted Class")
            output2_bar = gr.Image(label="Class Probabilities (Bar Chart)")
            output2_pie = gr.Image(label="Class Probabilities (Pie Chart)")

        with gr.Column():
            gr.Markdown("### Model 3: Neural Network (PyTorch)")
            output3_text = gr.Textbox(label="Predicted Class")
            output3_bar = gr.Image(label="Class Probabilities (Bar Chart)")
            output3_pie = gr.Image(label="Class Probabilities (Pie Chart)")

    # Predict button to update all models at once
    predict_button = gr.Button("Predict")
    predict_button.click(predict_model1, inputs=[sepal_length, sepal_width, petal_length, petal_width],
                         outputs=[output1_text, output1_bar, output1_pie])
    predict_button.click(predict_model2, inputs=[sepal_length, sepal_width, petal_length, petal_width],
                         outputs=[output2_text, output2_bar, output2_pie])
    predict_button.click(predict_model3, inputs=[sepal_length, sepal_width, petal_length, petal_width],
                         outputs=[output3_text, output3_bar, output3_pie])

# Run the app
demo.launch()
