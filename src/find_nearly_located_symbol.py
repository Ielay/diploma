import sys
from copy import copy

from math import sqrt


def euclidean_dist(p, q):
    return sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2))


def md_s(c, p):
    # 1/2 of euclidean distance
    return 0.5 * euclidean_dist(c, p)


def vect_dist(vect):
    return sqrt(vect[0]*vect[0] + vect[1]*vect[1])


def cos_2_vectors(a, b):
    return (scal_mult(a, b) / (vect_dist(a) * vect_dist(b)))


def scal_mult(a, b):
    return a[0]*b[0] + a[1]*b[1]


def md_c(c):
    # md_c = min d(c,e), e from edges, d(c,e) - Euclidean distance
    edge = c.first_edge

    min_dst = sys.float_info.max
    while True:
        p1 = edge.origin.point
        p2 = edge.twin.origin.point

        sect_vec = (p2.x - p1.x, p2.y - p1.y)
        p1_c_vec = (c.x - p1.x, c.y - p1.y)
        p2_c_vec = (c.x - p2.x, c.y - p2.y)

        scal_1 = scal_mult(sect_vec, p1_c_vec)
        scal_2 = scal_mult(sect_vec, sect_vec)

        if scal_1 < 0:
            dist = vect_dist(p1_c_vec)
            if min_dst > dist:
                min_dst = dist
        elif scal_2 < scal_1:
            dist = vect_dist(p2_c_vec)
            if min_dst > dist:
                min_dst = dist
        else: # calc as dist to a straight line in which the sect is located
            b = scal_1 / scal_2
            c_b = (p1.x + b * sect_vec[0], p1.y + b * sect_vec[1])

            dist = vect_dist((c.x - c_b[0], c.y - c_b[1]))
            if min_dst > dist:
                min_dst = dist

        edge = edge.next

        if edge == c.first_edge:
            break

    return min_dst


# АЛГОРИТМ ПОИСКА «СОСЕДНИХ»
# СИМВОЛОВ СЛОВА
# АЛГОРИТМ 1
def find_nearly_located_symbol(point, neighbours_points_set):
    # alpha = 1.3
    # beta = 1.5
    alpha = 2.2
    beta = 1.8

    nearly_located_points_set = []

    c = copy(point)
    for p in neighbours_points_set:
        dist_y = abs(c.x - p.x)

        min_dist_to_edge = min(md_c(c), md_c(p))

        if dist_y < alpha * min_dist_to_edge: #\
                # and md_s(c, p) < beta * min_dist_to_edge:
            nearly_located_points_set.append(p)

    return nearly_located_points_set
