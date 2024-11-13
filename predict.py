import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

def capture_image(alpha = 1.2, beta = 50):

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    ret, frame = cap.read()

    if ret:
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


        img_path = "Prediction/captured_image.jpg"
        cv2.imwrite(img_path, adjusted_frame)
        print(f"Image captured and saved as {img_path}.")
    else:
        print("Error: Could not read frame.")
        img_path = None

    cap.release()
    cv2.destroyAllWindows()
    return img_path



def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():

    img_path = capture_image()
    if img_path:
        model = tf.keras.models.load_model('my_model2.h5')

        preprocessed_img = preprocess_image(img_path)

        prediction = model.predict(preprocessed_img)
        predicted_class = 'Class 1' if prediction[0][0] > 0.5 else 'Class 0'

        print(f'The model predicts: {predicted_class}')

    else:
        print("Failed to capture image.")

if __name__ == "__main__":
    main()