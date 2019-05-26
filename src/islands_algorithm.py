import copy

from collections import deque

water = 0
land = 255


def islands_algorithm(map):
    local_map = copy.deepcopy(map)

    points_sets = []
    stack = deque()

    for i in range(len(local_map)):
        for j in range(len(local_map[i])):

            if local_map[i][j] == land:
                stack.append((i, j))

                new_set = []

                while len(stack) != 0:
                    (I, J) = stack.pop()

                    if local_map[I][J] == land:
                        new_set.append((I, J))

                        if len(local_map[0]) >= J+1:
                            stack.append((I, J+1))
                        if 0 <= J-1:
                            stack.append((I, J-1))
                        if len(local_map) >= I+1:
                            stack.append((I+1, J))
                        if 0 <= I-1:
                            stack.append((I-1, J))

                        local_map[I][J] = water

                points_sets.append(new_set)

    return points_sets