# import RPi.GPIO as GPIO # Used for shunt measurements
import time

# GPIO.setmode(GPIO.BCM)  # Usar numeración BCM
# GPIO.setup(17, GPIO.OUT)
# print('GPIO activo')
# GPIO.output(17, GPIO.HIGH)  # Escribir un valor alto (3.3V)

# GPIO.output(17, GPIO.LOW)
# time.sleep(0.1)
# GPIO.output(17, GPIO.HIGH)

import cv2
from ultralytics import YOLO

img_name = 'img_20221116_051503'
models_list = ['best_yolov3n','best_yolov5nu','best_yolov6n','best_yolov8n']

for model_name in models_list:
    img_path = f'imgs/{img_name}.jpg'
    img = cv2.imread(img_path)
    # Cargamos el modelo
    model = YOLO(f"models/{model_name}.pt")
    results = model.predict(img_path)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > 0:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(img, str(round(score, 1)), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Guardamos la imagen
    cv2.imwrite(f'results/{img_name}_{model_name}.jpg', img)