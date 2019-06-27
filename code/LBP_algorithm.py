import cv2
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pickle
from scipy.spatial import distance

# pridruzujemo nas direktorij projekta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# iz direktorija u image_dir smjestamo direktorij sa slikama iz projekta
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')




width = 550
height = 550
id_labels = []
label_ids = {}


#fja potrebna za konvertovanje binarnih vrijednosti u decimalnu
#binarne vrijednosti zapisujemo u niz

def binary_to_dec(binNum):
    decNum = 0
    power = 0
    for i in binNum:
        decNum += 2 ** power * i
        power += 1
    return decNum

#fja za citanje slika

def read_images():

    current_id = 0
    paths = []


    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("jpg") or file.endswith("jpeg") or file.endswith("JPG"):

                path = os.path.join(root, file)
                label = os.path.basename(root)
                paths.append(path)
                # label je ime foldera (katte middelton)
                # path je path slike u tom labelu

                # print(label, path)
                # print(root)

                # if not label in label_ids:
                # label_ids[label] = current_id
                # id_labels.append(current_id)

                label_ids[current_id] = label
                current_id += 1
    with open("face-labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    return paths

#funkcija za racunanje lbp vrijednosti i kreiranje histograma za svaku sliku

def calculate_lbp(grid_x, grid_y):
    # uzeli smo standardne vrijednosti radiusa i broja susjeda za centralni piksel

    # radius = 1
    # neighbors = 8

    count = 0
    images_paths = read_images()
    histograms = []
    for path in images_paths:


        # kako nemamo brojčane vrijednosti slike, potrebno je da koristimo numpy array za predstavljanje slika,
        # koje prvo učitamo pomoću python image library PIL.

        pil_image = Image.open(path).convert("L")  # konverzija u grey image
        size = (550, 550)
        final_image = pil_image.resize(size, Image.ANTIALIAS)
        image_array = np.array(final_image, "uint8")


        # vrijednosti u kojima cuvamo regione lica
        faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

        for (x, y, w, h) in faces:

            face_region = image_array[x:x + w, y:y + h]
            lbp_face_region = np.zeros((w, h))

            # provjeravamo da li su zadane dozvoljene vrijednosti broja blokova

            if grid_x < 0 or grid_x > w:
                return ValueError("Invalid gridX passes to function")
            if grid_y < 0 or grid_y > h:
                return ValueError("Invalid gridX passes to function")

            # racunamo sirinu i duzinu bloka
            grid_width = int(w / grid_x)
            grid_height = int(h / grid_y)


            #krajnje vrijednosti kopiramo, jer za njih ne mozemo izracunati lbp vrijednost
            for px in range(0, w):
                for py in range(0, h):
                    lbp_face_region[0][py] = face_region[0][py]
                    lbp_face_region[px][0] = face_region[px][0]
                    lbp_face_region[w-1][py] = face_region[w-1][py]
                    lbp_face_region[px][h-1] = face_region[px][h-1]

            for px in range(1, w-1):
                for py in range(1, h-1):


                    treshold = face_region[px][py]

                    # provjeravamo susjede centralnog piksela
                    if face_region[px - 1][py - 1] > treshold:
                        face_region[px - 1][py - 1] = 1
                    else:
                        face_region[px - 1][py - 1] = 0
                    if face_region[px + 1][py - 1] > treshold:
                        face_region[px + 1][py - 1] = 1
                    else:
                        face_region[px + 1][py - 1] = 0
                    if face_region[px][py - 1] > treshold:
                        face_region[px][py - 1] = 1
                    else:
                        face_region[px][py - 1] = 0
                    if face_region[px][py + 1] > treshold:
                        face_region[px][py + 1] = 1
                    else:
                        face_region[px][py + 1] = 0
                    if face_region[px + 1][py + 1] > treshold:
                        face_region[px + 1][py + 1] = 1
                    else:
                        face_region[px + 1][py + 1] = 0
                    if face_region[px + 1][py] > treshold:
                        face_region[px + 1][py] = 1
                    else:
                        face_region[px + 1][py] = 0
                    if face_region[px - 1][py + 1] > treshold:
                        face_region[px - 1][py + 1] = 1
                    else:
                        face_region[px - 1][py + 1] = 0
                    if face_region[px - 1][py] > treshold:
                        face_region[px - 1][py] = 1
                    else:
                        face_region[px - 1][py] = 0

                    # vrijednost tresholda u binarnom formatu
                    central_el = [face_region[px - 1][py - 1], face_region[px][py - 1],
                                  face_region[px + 1][py - 1],
                                  face_region[px - 1][py], face_region[px + 1][py],
                                  face_region[px - 1][py + 1], face_region[px][py + 1],
                                  face_region[px + 1][py + 1]]

                    decimal_central_element = binary_to_dec(central_el)

                    lbp_face_region[px][py] = decimal_central_element
            #print(lbp_face_region)


            # histogram jedne slike sastavljan od blokova
            histogram = []

            #prolazimo kroz broj zadanih gridova
            for i in range(0, grid_x):
                for j in range(0, grid_y):

                    if i == grid_x - 1:
                        end_position_x = len(lbp_face_region)
                    if j == grid_y - 1:
                        end_position_y = len(lbp_face_region[0])

                    start_position_x = i * grid_width
                    start_position_y = j * grid_height
                    end_position_x = (i + 1) * grid_width
                    end_position_y = (j + 1) * grid_height

                    # kreiramo histogram jednog bloka
                    hist_blok = np.zeros(256, "uint8")

                    for px in range(start_position_x, end_position_x):
                        for py in range(start_position_y, end_position_y):
                            hist_blok[int(lbp_face_region[px][py])] += 1

                    #kreiramo jedan histogram
                    histogram.extend(hist_blok)

            histograms.append(histogram)

    #print(len(histograms))
    #histogrami svake slike
    return histograms


def compare_histograms(hist1, hist2):

    dst = distance.euclidean(hist1, hist2)

    return dst

def load_face_and_identify():


    # čitamo podatke rječnika iz spašenog fajla i uradimo inverz, da prvo imamo id a zatim naziv labela
    # tj. label_ids[label] = current_id (jednom labelu jedan id, da je obratno bilo bi svaka slika ima svoj id ),
    #  treba nam obratno zbog uzimanja imena osobe.

    with open("face-labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        # labels = {k: v for k, v in og_labels.items()}
    #print(labels)
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)

            color = (255, 0, 0)  # BGR 0-255
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

            img_item = "pic.png"
            cv2.imwrite(img_item, roi_gray)

            # u lbp_face smjestamo lpb_vrijednosti testnog lica
            lbp_face = np.zeros((w, h))
            test_histogram = []
            for i in range(1, w-1):
                for j in range(1, h-1):

                    treshold = roi_gray[i][j]

                    # provjeravamo susjede centralnog piksela
                    if roi_gray[i - 1][j - 1] > treshold:
                        roi_gray[i - 1][j - 1] = 1
                    else:
                        roi_gray[i - 1][j - 1] = 0
                    if roi_gray[i + 1][j - 1] > treshold:
                        roi_gray[i + 1][j - 1] = 1
                    else:
                        roi_gray[i + 1][j - 1] = 0
                    if roi_gray[i][j - 1] > treshold:
                        roi_gray[i][j - 1] = 1
                    else:
                        roi_gray[i][j - 1] = 0
                    if roi_gray[i][j + 1] > treshold:
                        roi_gray[i][j + 1] = 1
                    else:
                        roi_gray[i][j + 1] = 0
                    if roi_gray[i + 1][j + 1] > treshold:
                        roi_gray[i + 1][j + 1] = 1
                    else:
                        roi_gray[i + 1][j + 1] = 0
                    if roi_gray[i + 1][j] > treshold:
                        roi_gray[i + 1][j] = 1
                    else:
                        roi_gray[i + 1][j] = 0
                    if roi_gray[i - 1][j + 1] > treshold:
                        roi_gray[i - 1][j + 1] = 1
                    else:
                        roi_gray[i - 1][j + 1] = 0
                    if roi_gray[i - 1][j] > treshold:
                        roi_gray[i - 1][j] = 1
                    else:
                        roi_gray[i - 1][j] = 0

                    #element u binarnom formatu
                    central_el = [roi_gray[i - 1][j - 1], roi_gray[i][j - 1],
                                  roi_gray[i + 1][j - 1],
                                  roi_gray[i - 1][j], roi_gray[i + 1][j],
                                  roi_gray[i - 1][j + 1], roi_gray[i][j + 1],
                                  roi_gray[i + 1][j + 1]]

                    decimal_central_element = binary_to_dec(central_el)

                    lbp_face[i][j] = decimal_central_element
            grid_x = 8
            grid_y = 8
            grid_width = int(len(lbp_face)/grid_x)
            grid_height = int(len(lbp_face[0])/grid_y)

            #kreiramo histogram testne slike
            for k in range(0, grid_x):
                for l in range(0, grid_y):

                    if k == grid_x - 1:
                        end_position_x = len(lbp_face)
                    if l == grid_y - 1:
                        end_position_y = len(lbp_face[0])

                    start_position_x = k * grid_width
                    start_position_y = l * grid_height
                    end_position_x = (k + 1) * grid_width
                    end_position_y = (l + 1) * grid_height

                    # kreiramo histogram jednog bloka
                    hist_blok = np.zeros(256, "uint8")

                    for px in range(start_position_x, end_position_x):
                        for py in range(start_position_y, end_position_y):
                            hist_blok[int(lbp_face[px][py])] += 1

                    # kreiramo jedan histogram
                    test_histogram.extend(hist_blok)


            #ucitamo trenirana lica i svako uporedimo sa testnim
            all_histograms = calculate_lbp(8, 8)
            id_ = 0
            min_d = 10000000
            for i in range(0, len(all_histograms)):

                d = compare_histograms(test_histogram, all_histograms[i])

                if min_d > d:
                    min_d = d
                    #cuvamo id najmanje razike kako bismo znali ciji je histogram
                    id_ = i

            #uzimamo ime po najmanjem id-u, odnosno najmanjoj razlici
            name = og_labels[id_]
            print(name)

        cv2.imshow('frame', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()




def main():
    load_face_and_identify()

main()
