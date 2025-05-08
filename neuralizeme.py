#!./venv/bin/python3

import pandas as pd
import matplotlib.pyplot as plt

import SimpleNeuralNetwork

DATA_ROUTE = './data/spiral.csv'


print("Init!")

snn = SimpleNeuralNetwork.SimpleNeuralNetwork()

# Leemos los datos de prueba.
full_dataFrame = pd.read_csv(filepath_or_buffer= DATA_ROUTE)
df  =  full_dataFrame[full_dataFrame['color'] == 'Red']
#print (f"Black : {dataFrame[dataFrame['color'] == 'Black']['x']}")

plt.scatter(
    df['x'], 
    df['y'], 
    c= df['color'],
    s= 40,
    cmap = plt.cm.Spectral
)
plt.show()

print("End!")