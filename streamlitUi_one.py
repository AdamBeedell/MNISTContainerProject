## streamlitUi One.py


##pip install streamlit

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as NN
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
import psycopg2


### logging

DB_PARAMS = {
    "dbname": "mnist_logs",
    "user": "postgres",
    "password": "helterskelt34", 
    "host": "postgres",   #### switch to localhost during testing
    "port": "5432"
}


def logthelogs(predicted_digit, confidence, actual):
    """Logs a prediction and actual into the DB"""
    try:
        dbconnection = psycopg2.connect(**DB_PARAMS) ## connect to DB
        cursor = dbconnection.cursor() 

        cursor.execute(
            "INSERT INTO predictions (predicted_digit, confidence, actual) VALUES (%s, %s, %s)" 
            ,(predicted_digit, confidence, actual)
        )

        dbconnection.commit()

        #tidyup
        cursor.close()
        dbconnection.close()

    except Exception as e:
        st.error(f"database error: {e}") ## error to streamlit


def clear_actual_input(): 
  st.session_state["actual_input"] = ''  # add "text" as a key using the square brackets notation and set it to have the value '' 


#### load trained model

### if GPU available, use that, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### recreate NN structure

class MNISTModel(NN.Module):   ### This creates a class for our specific NN, inheriting from the pytorch equivalent
    def __init__(self):  
        super().__init__()  ## super goes up one level to the torch NN module, and initializes the net
        self.fc1 = NN.Linear(28 * 28, 256)  # First hidden layer (784 pixel slots, gradually reducing down)
        self.fc2 = NN.Linear(256, 128)  # half as many nodes
        self.fc3 = NN.Linear(128, 64)   # half as many nodes
        self.fc4 = NN.Linear(64, 10) # Output layer (64 -> 10, one for each valid prediction)

    def forward(self, x):  # feed forward
        x = x.view(-1, 28 * 28)  # Flatten input from (batch, 1, 28, 28) -> (batch, 784), applies to the tensor prepared above in the dataloader
        x = F.relu(self.fc1(x))  # Activation function (ReLU), no negatives, play with leaky ReLU later
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation here, end of the road ("cross-entropy expects raw logits" - which are produced here, the logits will be converted to probabilities later by the cross-entropy function during training and softmax during training and inference)
        return x


model = MNISTModel() ## init
model.load_state_dict(torch.load("mnist_model_one_weights.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()  ## inference mode

transform = transforms.Compose([   ### same as during training
    transforms.Grayscale(num_output_channels=1), ### BW
    transforms.Resize((28, 28)), ### 28px square rez
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,)) ### -1 to 1
])

st.title("MNIST Digit Classifier")






col1, col2 = st.columns([3, 1])
with col1:
    st.write("Draw a digit below and get a prediction!")
    # Create a drawing canvas
    canvas_result = st_canvas(
        fill_color="black",  # Background color
        stroke_color="white",  # Draw in white (digits are typically white on black)
        stroke_width=10,  # Thickness of the drawn lines
        background_color="black",  # Black background
        height=280,  # Canvas size
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

with col2:
    st.write("**Correct Label:**")
    actual = st.text_input("Enter the true digit (0-9)", key="actual_input").strip()   ### this and beloy trying to keep it to valid digits in the DB
    actual = int(actual) if actual.isdigit() and 0 <= int(actual) <= 9 else None

    submit_actual = st.button("Submit Label", on_click=clear_actual_input)

# Process the drawn image
if canvas_result.image_data is not None:
    image = Image.fromarray(canvas_result.image_data.astype("uint8"))  # Convert to PIL image
    image = image.convert("L")  # Convert to grayscale
    
    #st.image(image, caption="Your Drawing", width=150)   ### unhash this to show the drawing back

    # Apply the same transformation as training
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    #Do inference
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_digit = torch.argmax(probabilities).item()
        confidence = torch.max(probabilities).item()

    # Display results
    st.write(f"**Prediction:** {predicted_digit}")
    st.write(f"**Confidence:** {confidence:.2%}")

    #log it
    logthelogs(predicted_digit, confidence, actual)


# Upload method
#canvas = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

#if canvas:
#    image = Image.open(canvas).convert("L")  # Convert to grayscale
#    image = ImageOps.invert(image)  # Invert colors (white digit on black background)
    
#    st.image(image, caption="Uploaded Digit", width=150)

    # Apply the transformation
#    image = transform(image)
#    image = image.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)
#    image = image.to(device)  #GPU to CPU Switch

    # Run inference
 #   with torch.no_grad():
 #       output = model(image)
 #       probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
 #       predicted_digit = torch.argmax(probabilities).item()
 #       confidence = torch.max(probabilities).item()

    # Display results
 #   st.write(f"**Prediction:** {predicted_digit}")
 #   st.write(f"**Confidence:** {confidence:.2%}")

if submit_actual:
    logthelogs(predicted_digit, confidence, actual)




