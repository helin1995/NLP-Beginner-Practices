import csv
import torch
import numpy as np
def load_data(data_path):
    data = []
    label = []
    with open(data_path, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            l, d = row
            data.append(d)
            label.append(int(l)-1)
    return data, label

def evaluate(model, criterion, X_dev, y_dev):
    dev_loss = 0.
    dev_correct = 0
    for x, y in zip(X_dev, y_dev):
        x = torch.from_numpy(x).type(dtype=torch.FloatTensor)
        y = torch.from_numpy(np.array([y])).type(dtype=torch.LongTensor)
        y_ = model(x)
        loss = criterion(y_, y)
        dev_loss += loss.detach().item()
        y_pred = torch.argmax(y_)
        if y_pred == y:
            dev_correct += 1
    return dev_loss / len(X_dev), dev_correct / len(X_dev)