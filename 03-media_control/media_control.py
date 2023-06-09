import cv2
import numpy as np
import config as c
from pynput.keyboard import Key, Controller
import sys
import keras
import time
from train_model import preprocess_image

def main():
    label_names = c.CONDITIONS
    label_names.append("no_gesture")

    # load the model
    model = keras.models.load_model("gesture_recognition")

    video_id = 0

    if len(sys.argv) > 1:
        video_id = int(sys.argv[1])

    # Create a video capture object for the webcam
    cap = cv2.VideoCapture(video_id)

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        img = preprocess_image(frame)
        img = np.array(img).astype('float32')
        img = img / 255.
        img = img.reshape(-1, c.IMG_SIZE, c.IMG_SIZE, c.COLOR_CHANNELS)

        prediction = model.predict(img)
        print(label_names[np.argmax(prediction)], np.max(prediction))

        keyboard = Controller()
        gesture = label_names[np.argmax(prediction)]
        if gesture == "like":
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
        elif gesture == "dislike":
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
        elif gesture == "stop":
            keyboard.press(Key.media_play_pause)
            keyboard.release(Key.media_play_pause)


        # Display the frame
        #cv2.imshow('frame', frame)
        time.sleep(0.3)

        # Wait for a key press and check if it's the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()