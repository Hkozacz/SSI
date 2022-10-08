import random
from typing import Optional

import matplotlib.pyplot as plt
from numpy import array, array_equal, genfromtxt, mean, ndarray
from scipy.spatial import distance


def get_random_m_samples(m: int, data: ndarray) -> list[ndarray]:
    return [data[random.randint(0, len(data) - 1)] for i in range(0, m)]


def get_distance(point: ndarray[float, float], center: ndarray[float, float]) -> float:
    return distance.cdist([point], [center]).min(axis=1, initial=None)


def group_data_by_center_points(data: ndarray, center_points: list[ndarray]) -> dict:
    groups: dict = {key: [] for key in range(0, len(center_points))}
    for point in data:
        distance_from_center = []
        for center_point in center_points:
            distance_from_center.append(get_distance(point, center_point))
        min_distance = min(distance_from_center)
        groups[distance_from_center.index(min_distance)].append(point)
    return groups


def hand_break(prev_center_points: ndarray, current_center_points: ndarray) -> bool:
    return array_equal(prev_center_points, current_center_points)


def k_mean(
    data: ndarray,
    m: int,
    max_iter: Optional[int] = None,
):
    center_points = get_random_m_samples(m, data)
    iter_number = 0
    while True:
        groups = group_data_by_center_points(data, center_points)
        new_center_points = []
        for group in groups.values():
            arr = array(group)
            plt.scatter(x=arr[:, 0], y=arr[:, 1])
            new_center_points.append([mean(arr[:, 0]), mean(arr[:, 1])])
        if hand_break(center_points, array(new_center_points)):
            print(f"Center points hasn't changed on {iter_number} iteration")
            break
        center_points = array(center_points)
        plt.scatter(x=center_points[:, 0], y=center_points[:, 1], c="black")
        center_points = array(new_center_points)
        plt.show()
        if max_iter and max_iter >= iter_number:
            break
        iter_number += 1


if __name__ == "__main__":
    opened_file = genfromtxt("spirala.txt", delimiter="   ")
    k_mean(opened_file, 10)
