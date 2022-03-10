import torch
import librosa
import pandas as pd
import numpy as np


import torch.nn as nn
import torch.nn.functional as F

import argparse


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 32), 1)
        self.conv2 = nn.Conv2d(16, 32, (1, 32), 1)
        self.pool = nn.MaxPool2d((1, 2), 2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(1763008, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = self.sigmoid(x)
        return output


def read_files(csv_path, audio_folder_path):
    track_df = pd.read_csv(csv_path)

    audio_dict = {
        "track": [],
        "y": [],
        "sr": [],
    }

    for track_name in track_df["track"]:
        print(f"loading {track_name}")
        y, sr = librosa.load(f"{audio_folder_path}/{track_name}")
        audio_dict["track"].append(track_name)
        audio_dict["y"].append(y)
        audio_dict["sr"].append(sr)

    return track_df, pd.DataFrame(audio_dict)

def read_files(filename):
    track_dict = {
        "track": filename,
    }
    audio_dict = {
        "track": [],
        "y": [],
        "sr": [],
    }

    track_df = pd.DataFrame(track_dict)

    for track_path in track_df["track"]:
        print(f"loading {track_path}")
        y, sr = librosa.load(track_path)
        audio_dict["track"].append(track_path)
        audio_dict["y"].append(y)
        audio_dict["sr"].append(sr)

    return track_df, pd.DataFrame(audio_dict)



def preprocess_data(track_df, audio_df):
    x = np.array([[[value]] for value in audio_df["y"].values])
    return x


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--csv", dest="csv", default="")
parser.add_argument("--filename", dest="filename", nargs='+', default="")
args = parser.parse_args()

model_folder_path = "../model"
data_folder_path = "../data"
load_model_name = "success_model4.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Model...")
model = torch.load(f"{model_folder_path}/{load_model_name}")
model.eval()


output_dict = {
    "track": [],
    "score": []
}

if(args.csv != ""):
    test_track_df, test_audio_df = read_files(args.csv, f"{data_folder_path}/audios/clips")
elif(args.filename != ""):
    test_track_df, test_audio_df = read_files(args.filename)

test_x = preprocess_data(test_track_df, test_audio_df)
for track, features in zip(test_track_df['track'], test_x):
    features = np.array([features])
    features = torch.tensor(features, dtype=torch.float32).to(device)
    score = model(features)
    output_dict["track"].append(track)
    output_dict["score"].append(score[0][0].cpu().detach().numpy())

output_df = pd.DataFrame(output_dict)
output_df.to_csv(f"submission.csv", index=False)
