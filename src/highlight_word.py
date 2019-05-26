from collections import deque

from src.find_nearly_located_symbol import find_nearly_located_symbol


def get_neighbours(point, points_with_neighbours):
    for (p, neighbours_set) in points_with_neighbours:
        if p == point:
            return neighbours_set

    return None


def highlight_word(the_point, points_with_neighbours):
    symbols_of_word = [the_point]

    points_stack = deque()
    points_stack.append(the_point)

    while len(points_stack) > 0:
        point = points_stack.pop()
        neighbours_points = get_neighbours(point, points_with_neighbours)

        one_word_points = find_nearly_located_symbol(point, neighbours_points)

        for pnt in one_word_points:
            if pnt not in symbols_of_word:
                symbols_of_word.append(pnt)
                points_stack.append(pnt)

    return symbols_of_word
