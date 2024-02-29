import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# Carga el modelo TFLite
model_path='models/con-int8/best_yolov3n_saved_model/best_yolov3n_int8.tflite'
interpreter = tflite.Interpreter(model_path,
  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Dimensiones de entrada esperadas por el modelo
input_details = interpreter.get_input_details()
input_height, input_width = input_details[0]['shape'][1], input_details[0]['shape'][2]

# Carga de la imagen y preprocesamiento
image = Image.open('imgs/img_20221116_051503.jpg').resize((input_width, input_height))
input_data = np.expand_dims(image, axis=0)
input_data = input_data / 255.0  # Normaliza los píxeles

# Asigna los datos de entrada
interpreter.set_tensor(input_details[0]['index'], input_data)

# Ejecuta la inferencia
interpreter.invoke()

# Obtiene los resultados de la inferencia
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
