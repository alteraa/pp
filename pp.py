import math
import numpy as np
import matplotlib.pyplot as plt

def is_point_in_polygon(point, polygon):
    intersections = 0
    x, y = point
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        if ((p1[1] > y) != (p2[1] > y)) and (x < (p2[0] - p1[0]) * (y - p1[1]) / (p2[1] - p1[1]) + p1[0]):
            intersections += 1
    return intersections % 2 == 1

def get_closest_point_on_polygon(point, polygon):
    closest_point = None
    min_distance = math.inf
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        segment_length_squared = (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
        if segment_length_squared == 0:
            distance_squared = (point[0] - p1[0])**2 + (point[1] - p1[1])**2
        else:
            t = ((point[0] - p1[0]) * (p2[0] - p1[0]) + (point[1] - p1[1]) * (p2[1] - p1[1])) / segment_length_squared
            t = max(0, min(1, t))
            closest_point_on_segment = (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))
            distance_squared = (point[0] - closest_point_on_segment[0])**2 + (point[1] - closest_point_on_segment[1])**2
        if distance_squared < min_distance:
            closest_point = closest_point_on_segment
            min_distance = distance_squared
    return closest_point

def get_point_inside_polygon_with_offset(point, polygon, offset):
    closest_point = get_closest_point_on_polygon(point, polygon)
    direction = (closest_point[0] - point[0], closest_point[1] - point[1])
    length = math.sqrt(direction[0]**2 + direction[1]**2)
    unit_direction = (direction[0] / length, direction[1] / length)
    new_point = (closest_point[0] + unit_direction[0] * offset, closest_point[1] + unit_direction[1] * offset)
    if is_point_in_polygon(new_point, polygon):
        return new_point
    else:
        return get_point_inside_polygon_with_offset(new_point, polygon, offset)

def rand_vals(min_, max_, len_=1000):
    xmin, xmax = min_[0], max_[0]
    ymin, ymax = min_[1], max_[1]
    rand = np.random.RandomState(42)
    xvals = rand.uniform(xmin, xmax, len_)
    yvals = rand.uniform(ymin, ymax, len_)
    return xvals, yvals

def plot_dots(xvals, yvals, colors=None):
    assert len(xvals) == len(yvals)
    ax = plt.subplot(111)
    if colors is None:
        colors = 'red'
    ax.scatter(xvals, yvals, c=colors, s=50, marker='.')

def plot_polygon(polygon):
    ax = plt.subplot(111)
    xvals, yvals = [], []
    for p in polygon:
        xvals.append(p[0])
        yvals.append(p[1])
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.plot(xvals, yvals, 'b^--')

def rand_vals(min_, max_, len_=1000):
    xmin, xmax = min_[0], max_[0]
    ymin, ymax = min_[1], max_[1]
    rand = np.random.RandomState(42)
    xvals = rand.uniform(xmin, xmax, len_)
    yvals = rand.uniform(ymin, ymax, len_)
    return xvals, yvals

if __name__ == '__main__':
    # point = (2.5, 1.68)
    polygon = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        # (0, 0),
    ]
    # print(is_point_in_polygon(point, polygon))
    # print(get_closest_point_on_polygon(point, polygon))
    xv, yv = rand_vals((-1, -1), (2, 2), 1000)
    for i, (x, y) in enumerate(zip(xv.copy(), yv.copy())):
        # x_, y_ = get_closest_point_on_polygon((x, y), polygon)
        x_, y_ = get_point_inside_polygon_with_offset((x, y), polygon, 0.1)
        xv[i] = x_
        yv[i] = y_
    plot_polygon(polygon)
    plot_dots(xv, yv)
    plt.show()
