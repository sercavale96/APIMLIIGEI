from fastapi import FastAPI
import pandas as pd
import pickle
import numpy as np

with open ('Models/modelo.pkl' , 'rb') as m: # En modo lectura
            modelo = pickle.load(m)

app = FastAPI()

@app.get('/')
def hello():
    return {'message':'Hello World'}

@app.post('/predict')
def predict(request: dict):
    data = request['data']
    input_data = np.array(data)  # Convertir la lista en un arreglo NumPy
    input_data = input_data.reshape(1, -1)  # Convertir el arreglo 1D en una matriz 2D
    prediction = modelo.predict(input_data)
    output = int(prediction[0])  # Convertir el valor de la predicción a un entero
    return {'prediction': output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)