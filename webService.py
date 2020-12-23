#from EmergingTechProject.ipynb

# Neural networks.
import tensorflow.keras as kr

# Numerical arrays
import numpy as np

# Data frames.
import pandas as pd

# Plotting
import matplotlib.pyplot as plt

# Plot style.
plt.style.use("ggplot")

# Plot size.
plt.rcParams['figure.figsize'] = [14, 8]

powerData = pd.read_csv("powerproduction.csv")

# Build our model.
model = kr.models.Sequential()
model.add(kr.layers.Dense(50, input_shape=(1,), activation='sigmoid', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
model.add(kr.layers.Dense(1, activation='linear', kernel_initializer="glorot_uniform", bias_initializer="glorot_uniform"))
model.compile('adam', loss='mean_squared_error')

model.fit(powerData['speed'], powerData['power'], epochs=1000, batch_size=10)


user = int(input("Enter STUFF NOW PLS: "));
powerData['speed'] = user
powerData['power'] = model.predict(powerData['speed'])
prediction = powerData['power'][0]
print("Prediction: ",prediction)