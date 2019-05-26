from copy import copy


def find_nearly_area(the_point, other_points): # centers of mass of symbols
    coords = the_point.get_coordinates()
    other_points_local = copy(other_points)
    other_points_local.remove(the_point)

    nearly_areas = []

    for other_point in other_points_local:
        other_coords = other_point.get_coordinates()
        matches_numb = 0

        for oth_coord in other_coords:
            if oth_coord in coords:
                matches_numb += 1

        if matches_numb >= 2:
            nearly_areas.append(other_point)

    return nearly_areas
