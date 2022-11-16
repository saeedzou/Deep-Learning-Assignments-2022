import torch
import cv2
import numpy as np
from torch import nn
import torch.nn.functional as F
import string

# load alphabet into a list and create a dictionary to map each letter to a number
alphabet = list(string.ascii_uppercase)[:25]
num2letter = {num: letter for num, letter in enumerate(alphabet)}

class MLP(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# initialize the model and load model_Adam.pt from current directory
model = MLP(784, 256, 25)
model.load_state_dict(torch.load('model_Adam.pt'))
model.eval()

# define a function that turns on the webcam and captures an image
# then preprocess the image as a tensor of size 1x784
# and pass it to the model to get the prediction
# print the prediction on the webcam window
# press q to quit
# you can use cv2.putText to print the prediction on the webcam window
# you can use cv2.imshow to show the webcam window
# you can use cv2.VideoCapture to turn on the webcam
# you can use cv2.waitKey to wait for a key press
# you can use cv2.destroyAllWindows to close the webcam window
# you can use cv2.resize to resize the image
# you can use cv2.cvtColor to convert the image to grayscale
#

def predict():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("failed to grab frame")
            break
        x = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x = cv2.resize(x, (28, 28))
        x = np.array(x).reshape(1, -1)
        x = torch.from_numpy(x).float()
        pred = model(x)
        pred = pred.argmax(dim=1, keepdim=True)
        print(pred)
        cv2.putText(frame, str(num2letter[pred.item()]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()   

predict()

