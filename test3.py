import argparse
import math
import os
import random
import urllib.request
import urllib.error
import pathlib
import numpy
import cv2
from sklearn import model_selection
from keras import models, layers

def make_model():
    model = models.Sequential()
    model.add(layers.Conv2D(4, kernel_size=3, activation="relu", input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(4, kernel_size=3, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Conv2D(8, kernel_size=3, activation="relu", input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(8, kernel_size=3, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Conv2D(4, kernel_size=3, activation="relu", input_shape=(64, 64, 3)))
    model.add(layers.Conv2D(4, kernel_size=3, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(50, activation="relu"))
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# select galaxies
ell_string = ""
sp_string = ""
edge_string = ""
with open("../GalaxyZoo1_DR_table2.csv", "r") as inp:
    next(inp)
    for s in inp:
        vals = s.split(",")
        ra = vals[1].split(":")
        dec = vals[2].split(":")
        if float(vals[4]) > 0.9:
            ell_string = ell_string + f"nearest:B {ra[0]} {ra[1]} {ra[2]} {dec[0]} {dec[1]} {dec[2]}\n"
        elif (float(vals[5]) > 0.75 or float(vals[6]) > 0.75):
            sp_string = sp_string + f"nearest:B {ra[0]} {ra[1]} {ra[2]} {dec[0]} {dec[1]} {dec[2]}\n"
        elif float(vals[7]) > 0.8:
            edge_string = edge_string + f"nearest:B {ra[0]} {ra[1]} {ra[2]} {dec[0]} {dec[1]} {dec[2]}\n"
file = open("elliptical.dat", "w")
file.write(ell_string)
file.close()
file = open("spiral.dat", "w")
file.write(sp_string)
file.close()
file = open("edge.dat", "w")
file.write(edge_string)
file.close()

#filter galaxies
ell_string = ""
sp_string = ""
edge_string = ""
with open("elliptical.dat", "r") as file:
    for line in file:
        if line.startswith("nearest"):
            vals = s.split("|")
            if not vals[4].isspace():
                d = math.pow(10, float(vals[4])) * 6
                if d_min < d:
                    coord = vals[0].split()
                    ell_string = ell_string + f"{coord[1]} {coord[2]} {coord[3]} {coord[4]} {coord[5]} {coord[6]} {vals[1]} {d}\n"
with open("spiral.dat", "r") as file:
    for line in file:
        if line.startswith("nearest"):
            vals = s.split("|")
            if not vals[4].isspace():
                d = math.pow(10, float(vals[4])) * 6
                if d_min < d:
                    coord = vals[0].split()
                    sp_string = sp_string + f"{coord[1]} {coord[2]} {coord[3]} {coord[4]} {coord[5]} {coord[6]} {vals[1]} {d}\n"
with open("edge.dat", "r") as file:
    for line in file:
        if line.startswith("nearest"):
            vals = s.split("|")
            if not vals[4].isspace():
                d = math.pow(10, float(vals[4])) * 6
                if d_min < d:
                    coord = vals[0].split()
                    edge_string = edge_string + f"{coord[1]} {coord[2]} {coord[3]} {coord[4]} {coord[5]} {coord[6]} {vals[1]} {d}\n"
file = open("elliptical.dat", "w")
file.write(ell_string)
file.close()
file = open("spiral.dat", "w")
file.write(sp_string)
file.close()
file = open("edge.dat", "w")
file.write(edge_string)
file.close()

#download images
size = 64
scale = 1
with open("elliptical.dat", "r") as file:
    i = 0
    for line in file:
        vals = line.split()
        ra = f"{vals[0]}:{vals[1]}:{vals[2]}"
        dec = f"{vals[3]}:{vals[4]}:{vals[5]}"
        url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}"
        urllib.request.urlretrieve(url, f"images/elliptical_{i}.jpg")
        i += 1
with open("spiral.dat", "r") as file:
    i = 0
    for line in file:
        vals = line.split()
        ra = f"{vals[0]}:{vals[1]}:{vals[2]}"
        dec = f"{vals[3]}:{vals[4]}:{vals[5]}"
        url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}"
        urllib.request.urlretrieve(url, f"images/spiral_{i}.jpg")
        i += 1
with open("edge.dat", "r") as file:
    i = 0
    for line in file:
        vals = line.split()
        ra = f"{vals[0]}:{vals[1]}:{vals[2]}"
        dec = f"{vals[3]}:{vals[4]}:{vals[5]}"
        url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={size}&height={size}"
        urllib.request.urlretrieve(url, f"images/edge_{i}.jpg")
        i += 1

# train and make model
elliptical_set = []
spiral_set = []
edge_set = []
for f in pathlib.Path("images/elliptical_").glob("*.jpg"):
    filename = str(f)
    elliptical_set.append(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
for f in pathlib.Path("images/spiral_").glob("*.jpg"):
    filename = str(f)
    spiral_set.append(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
for f in pathlib.Path("images/edge_").glob("*.jpg"):
    filename = str(f)
    edge_set.append(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
n_ell = len(elliptical_set)
n_sp = len(spiral_set)
n_edge = len(edge_set)
n_per_type = 3 * max(n_ell, n_sp, n_edge)
for i in range(n_ell, n_per_type):
    m = cv2.getRotationMatrix2D((random.randint(15, 70), random.randint(15, 70)), random.randint(-180, 180), 1)
    elliptical_set.append(cv2.warpAffine(elliptical_set[random.randint(0, n_ell)], m, (64, 64)))
for i in range(n_sp, n_per_type):
    m = cv2.getRotationMatrix2D((random.randint(15, 70), random.randint(15, 70)), random.randint(-180, 180), 1)
    spiral_set.append(cv2.warpAffine(spiral_set[random.randint(0, n_sp)], m, (64, 64)))
for i in range(n_edge, n_per_type):
    m = cv2.getRotationMatrix2D((random.randint(15, 70), random.randint(15, 70)), random.randint(-180, 180), 1)
    edge_set.append(cv2.warpAffine(edge_set[random.randint(0, n_edge)], m, (64, 64)))
galaxies = elliptical_set + spiral_set + edge_set
labels = []
for i in range(n_per_type):
    labels.append(numpy.array([1, 0, 0]))
for i in range(n_per_type, 2 * n_per_type):
    labels.append(numpy.array([0, 1, 0]))
for i in range(2 * n_per_type, 3 * n_per_type):
    labels.append(numpy.array([0, 0, 1]))
galaxies_array = numpy.array(galaxies)
labels_array = numpy.array(labels)
data_train, data_test, labels_train, labels_test = model_selection.train_test_split(galaxies_array, labels_array, test_size=0.2)
model = make_model()
model.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=5, batch_size=32)
model.save("galaxy_identification_model.h5")

parser = argparse.ArgumentParser()
#parser.add_argument("--make_galaxy_list", action="store_true")
#parser.add_argument("--make_set", action="store_true")
#parser.add_argument("--download_images", action="store_true")
#parser.add_argument("--train_model", action="store_true")
parser.add_argument("--predict", nargs="+", action="store")
args = parser.parse_args()
str_results = ["elliptical", "spiral", "edge-on"]
#if args.make_galaxy_list:
#    select_galaxies()
#    exit()
#if args.make_set:
#    filter_galaxies(15)
#    exit()
#if args.download_images:
#    download_images()
#    exit()
#if args.train_model:
#    galaxies_array, labels_array = make_dataset()
#    data_train, data_test, labels_train, labels_test = model_selection.train_test_split(galaxies_array, labels_array, test_size=0.2)
#    model = make_model()
#    model.fit(data_train, labels_train, validation_data=(data_test, labels_test), epochs=5, batch_size=32)
#    model.save("classifier.h5")
#    exit()
if len(args.predict) > 0:
    model = models.load_model("classifier.h5")
    for f in args.predict:
        img = cv2.imread(f)
        if img.shape[0] < img.shape[1]:
            center = round(img.shape[1] / 2)
            sz = round(img.shape[0] / 2)
            img = numpy.copy(img[:, (center - sz):(center + sz), :])
        else:
            center = round(img.shape[0] / 2)
            sz = round(img.shape[1] / 2)
            img = numpy.copy(img[(center - sz):(center + sz), :, :])
        img = cv2.cvtColor(cv2.resize(img, (64, 64)), cv2.COLOR_BGR2RGB)
        print(f"{f}:")
        res = model.predict(numpy.array([img]))
        print(f"elliptical: {res[0][0]}")
        print(f"spiral: {res[0][1]}")
        print(f"edge-on: {res[0][2]}")
        if max(res[0]) > 0.75:
            print(f"RESULT: {str_results[numpy.argmax(res)]} galaxy\n")
        else:
            print(f"Not classified")
