import cv2
import os
import numpy as np
from PIL import Image
import pickle

#pridruzujemo nas direktorij projekta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#iz direktorija u image_dir smjestamo direktorij sa slikama iz projekta
image_dir = os.path.join(BASE_DIR, "images")


current_id = 0
label_ids = {}
y_train = []
x_train = []

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

# def read_images():
#
#
#     paths = []
#     for root, dirs, files in os.walk(image_dir):
#         for file in files:
#             if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("JPG"):
#
#                 path = os.path.join(root, file)
#                 label = os.path.basename(root)
#                 paths.append(path)
#                 # label je ime foldera (katte middelton)
#                 # path je path slike u tom labelu
#
#                 # print(label, path)
#                 # print(root)
#
#                 if not label in label_ids:
#                     label_ids[label] = current_id
#                     current_id += 1
#                 id_ = label_ids[label]
#
#                 y_train.append(id_)
#     # pickle koristimo za spasavanje potadaka u binary formatu
#     with open("face-labels.pickle", 'wb') as f:
#         pickle.dump(label_ids, f)
#
#     return paths


def create_face_values():
    current_id = 0
    # data = []
    # paths = read_images()
    #
    # for path in paths:
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("JPG"):

                path = os.path.join(root, file)
                label = os.path.basename(root)

                # label je ime foldera (katte middelton)
                # path je path slike u tom labelu

                # print(label, path)
                # print(root)

                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]


                #kako nemamo brojčane vrijednosti slike, potrebno je da koristimo numpy array za predstavljanje slika,
                #koje prvo učitamo pomoću python image library PIL.

                pil_image = Image.open(path).convert("L") #konverzija u grey image
                size = (550, 550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(final_image, "uint8")



               #u faces čuvamo vrijednosti (pravougaonike) koji su detektovani kao lice
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_train.append(id_)

    # pickle koristimo za spasavanje potadaka u binary formatu
    with open("face-labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    return x_train


def train_data():

    data = create_face_values()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(data, np.array(y_train))
    recognizer.save("face-trainner.yml")



def capture_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("face-trainner.yml")

    # čitamo podatke rječnika iz spašenog fajla i uradimo inverz, da prvo imamo id a zatim naziv labela
    # tj. label_ids[label] = current_id (jednom labelu jedan id, da je obratno bilo bi svaka slika ima svoj id ), treba nam obratno zbog uzimanja imena osobe.

    with open("face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            roi_color = frame[y:y + h, x:x + w]

            id_, conf = recognizer.predict(roi_gray)
            if conf >= 50:
                # print(5: #id_)
                # print(labels[id_])
                # print(conf)
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name + ' ' + str(conf), (x, y), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "pic.png"
            cv2.imwrite(img_item, roi_color)

            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)



        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




def main():

    create_face_values()
    train_data()
    capture_face()


main()