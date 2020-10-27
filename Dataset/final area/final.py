import os
import numpy as np


path = './'
items = os.listdir(path)
targets = []
data = []
alpha = ['a','b','c','d','h','i','j','k']
for item in items:
    for i in range(len(alpha)):
        if item.find('.DS') != -1 :
            continue
        if item.find('.py') !=-1 :
            continue
        letter = alpha[i]
        if item.find(letter) != -1:
            targets.append(i+1)
            letter_data = np.load(path + item)
            data.append(letter_data)
targets = np.asarray(targets)
data = np.asarray(data)
np.save('data.npy', data)
np.save('labels.npy', targets)

