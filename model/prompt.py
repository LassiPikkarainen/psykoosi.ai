# load ascii text and covert to lowercase
filename = "combined.txt"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import random

seq_length = 70
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()


chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=512, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512, 58)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
 
model = CharModel()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# Generation using the trained model
best_model, char_to_int = torch.load("single-char.pth")
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)

print(f"Model loaded. Vocab size: {n_vocab}, Int to Char mapping: {int_to_char}")

 
model.eval()
    
while True:
    prompt = input(f"Enter a prompt ({seq_length} chars): ")
    prompt = prompt.lower()
    if len(prompt) < seq_length:
        rand_start = np.random.randint(0, len(raw_text) - (seq_length - len(prompt)))
        prompt = raw_text[rand_start:rand_start + seq_length - len(prompt)] + " " + prompt
        print(f"Prompt padded to {seq_length} chars: {prompt}")
        pattern = [char_to_int[c] for c in prompt if c in char_to_int]
        len_ = random.randint(200, 1000)
        with torch.no_grad():
            for i in range(len_):
                # format input array of int into PyTorch tensor
                x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
                x = torch.tensor(x, dtype=torch.float32)
                # generate logits as output from the model
                prediction = model(x.to(device))

                # get the predicted character as a one-hot encoded vector
                prediction_ = torch.softmax(prediction, dim=1).cpu().numpy().flatten()
                prob = prediction_ * prediction_
                prob = prob / np.sum(prob)
                index = np.random.choice(len(prediction_), p=prob)

                #index = int(prediction.argmax())

                #print(f"Predicted index: {index}, Predicted char: {int_to_char[index]}")
                result = int_to_char[index]
                #print(result)
                #prediction = torch.softmax(prediction, dim=1).cpu().numpy().flatten()
                #print(f"Prediction probabilities: {prediction}")
                print(result, end="")
                # append the new character into the prompt for the next iteration
                pattern.append(index)
                pattern = pattern[1:]
        print()
        print("Done.")