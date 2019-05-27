import sys
from collections import deque

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.engine.saving import load_model

from src.find_nearly_area import find_nearly_area
from src.highlight_word import highlight_word
from src.islands_algorithm import islands_algorithm

# voronoi
from voronoi.algorithm import Algorithm
from voronoi import BoundingBox

# ----------------------------------------------
signs_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,

    "a": 10,
    "b": 11,
    "c": 12,
    "d": 13,
    "e": 14,
    "f": 15,
    "g": 16,
    "h": 17,
    "i": 18,
    "j": 19,
    "k": 20,
    "l": 21,
    "m": 22,
    "n": 23,
    "o": 24,
    "p": 25,
    "q": 26,
    "r": 27,
    "s": 28,
    "t": 29,
    "u": 30,
    "v": 31,
    "w": 32,
    "x": 33,
    "y": 34,
    "z": 35,

    "less": 36,
    "less_or_eq": 37,
    "equal": 38,
    "greater": 39,
    "greater_or_eq": 40,
    "left_bracket": 41,
    "right_bracket": 42,
    "plus": 43,
    "minus": 44,
    "times": 45,
    "div": 46
}
# picture path
equation_path = '../equations/eq3.png'
# read image
image = cv2.imread(equation_path)
# convert color to gray
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 255 = white, 0 = black
# invert color (text is going from black to white)
image = cv2.bitwise_not(image)

threshold = 155

# filter (reduce low intensity pixels)
for i in range(0, len(image)):
    for j in range(0, len(image[0])):
        if image[i][j] < threshold:
            image[i][j] = 0
        else:
            image[i][j] = 255

# define a sets of points such that each set is a single digit from a picture
points_sets = islands_algorithm(image)

print("A number of digits:", len(points_sets))

# find each number center of mass
mass_center_to_points_set = dict()
mass_centers = []
for set in points_sets:
    i_sum = 0
    j_sum = 0

    for (i, j) in set:
        i_sum += i
        j_sum += j

    center = (i_sum // len(set), j_sum // len(set))

    mass_centers.append(center)
    mass_center_to_points_set[center] = set

# |y| diff
delta = (len(image) / 100) * 12

# |x| diff
eps = (len(image[0]) / 100) * 3

# remove 2 points in '=' and make a single point for it
new_mass_centers = []

removed_points = []
for i in range(len(mass_centers)):
    (y1, x1) = mass_centers[i]

    is_founded = False
    for j in range(i + 1, len(mass_centers)):
        (y2, x2) = mass_centers[j]

        if (abs(x1 - x2) < eps) and (abs(y1 - y2) < delta):
            # it's a equality sign
            new_x = (x1 + x2) // 2
            new_y = (y1 + y2) // 2

            new_mass_centers.append((new_y, new_x))

            # точки из двух множеств смержить в точки одного множества
            pnts_from = mass_center_to_points_set.pop(mass_centers[j])
            pnts_to = mass_center_to_points_set.pop(mass_centers[i])
            for pnt in pnts_from:
                pnts_to.append(pnt)
            mass_center_to_points_set[(new_y, new_x)] = pnts_to

            removed_points.append(j)
            is_founded = True

            break

    if not is_founded and i not in removed_points:
        new_mass_centers.append((y1, x1))

mass_centers = new_mass_centers

print("Mass centers after processing: ", mass_centers)

###############
# Voronoi algorithm
polygon = BoundingBox(0, len(image), 0, len(image[0]))

points = mass_centers

v = Algorithm(polygon)
v.create_diagram(points=points, vis_steps=False, verbose=False, vis_result=False, vis_tree=False, vis_before_clipping=False)

###############
points_with_bounding_vertices = v.points

# set a point with its graph neighbours
point_with_neighbours = []
for point in points_with_bounding_vertices:
    set = find_nearly_area(point, points_with_bounding_vertices)
    point_with_neighbours.append((point, set))

    print("Point:", point, "; neighbours set:", set)

st = deque()
for p_w_n in point_with_neighbours:
    st.append(p_w_n[0])

equations = []
while len(st) > 0:
    pnt = st.pop()

    res = highlight_word(pnt, point_with_neighbours)
    sorted_res = sorted(res, key=lambda pnt: pnt.y, reverse=False)

    equations.append(sorted_res)
    print(sorted_res)

    for i in sorted_res:
        if i in st:
            st.remove(i)
###########################
# Load nn model
model_name = "my_nn_convolution_model.h5"
nn_model = load_model(model_name)

equations_of_pic = []

cnt = 0
for math_eq in equations:
    eq_of_pic = []
    for math_sign in math_eq:
        cnt += 1

        cell_center = (math_sign.x, math_sign.y)

        sign_points_set = mass_center_to_points_set.get(cell_center)

        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = 0
        max_y = 0
        for (y_i, x_i) in sign_points_set:
            if x_i < min_x:
                min_x = x_i
            if y_i < min_y:
                min_y = y_i
            if x_i > max_x:
                max_x = x_i
            if y_i > max_y:
                max_y = y_i

        y_diff = (max_y - min_y) + 1
        x_diff = (max_x - min_x) + 1

        max_axis_val = max(y_diff, x_diff)
        min_axis_val = min(y_diff, x_diff)
        sign_arr = np.zeros(shape=(max_axis_val, max_axis_val), dtype=int)

        l = int(((max_axis_val - min_axis_val) + 1) / 2)

        for (y_i, x_i) in sign_points_set:
            if (y_diff > x_diff):
                # print("Max x:", max_x,
                #       "Max y:", max_y,
                #       "Min x:", min_x,
                #       "Min y:", min_y,
                #       "Max axis val:", max_axis_val,
                #       "Min axis val:", min_axis_val,
                #       "x:", x_i,
                #       "y:", y_i,
                #       "l:", l)
                sign_arr[(y_i - min_y)][(x_i - min_x) + l] = 255
            else: # if max_y <= max_x
                # print("Max x:", max_x,
                #       "Max y:", max_y,
                #       "Min x:", min_x,
                #       "Min y:", min_y,
                #       "Max axis val:", max_axis_val,
                #       "Min axis val:", min_axis_val,
                #       "x:", x_i,
                #       "y:", y_i,
                #       "l:", l)
                sign_arr[(y_i - min_y) + l][(x_i - min_x)] = 255

        # sign_arr = cv2.bitwise_not(sign_arr)
        for i in range(len(sign_arr)):
            for j in range(len(sign_arr[0])):
                sign_arr[i][j] = (255 - sign_arr[i][j])
        ##############
        cv2.imwrite("../images/" + math_sign.name + str(cnt) + ".jpg", sign_arr)

        img = cv2.imread("../images/" + math_sign.name + str(cnt) + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        resized_img = cv2.resize(img, dsize=(45, 45), interpolation=cv2.INTER_CUBIC)
        # save result
        cv2.imwrite(("../images/" + math_sign.name + str(cnt) + ".jpg"), resized_img)

        resized_img = cv2.bitwise_not(resized_img)
        ##############

        reshaped_final_img = np.array([np.reshape(np.array(resized_img), (45, 45, 1))])  # for convolution model

        # predict number
        answer = nn_model.predict(reshaped_final_img)

        index_of_max_val_in_answer = list(answer[0]).index(max(answer[0]))
        for sign, ordered_numb in signs_dict.items():
            if ordered_numb == index_of_max_val_in_answer:
                # print("Answer:", sign)
                eq_of_pic.append(sign)

                break

    equations_of_pic.append(eq_of_pic)

print(equations_of_pic)

