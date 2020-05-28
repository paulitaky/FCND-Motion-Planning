from scipy.spatial import distance
from shapely.geometry import Polygon, Point, LineString
from sklearn.neighbors import KDTree
import numpy.linalg as LA
import networkx as nx

from enum import Enum
from queue import PriorityQueue
import numpy as np
import math
import pkg_resources
pkg_resources.require("networkx==2.1")


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
            ]
            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1

    return grid, int(north_min), int(east_min)


# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)

    return valid_actions


def get_valid_graph_node_locations(random_graph_node_locations, 
                                   base_polygon_0,
                                   base_polygon_1,
                                   tree,
                                   obstacle_polygons, 
                                   no_of_neighbors,
                                   safety_distance):
    """ Gets valid (non obstacle colliding) locations for graph nodes within the input grid. """
    valid_graph_node_locations = []
    for point in random_graph_node_locations:
        # use k nearest neighbor polygons in case representative points
        # are that ill chosen that considered point is within one polygon
        # but is checked with other (closest) polygon
        if not collides(base_polygon_0,
                        base_polygon_1, 
                        tree, 
                        obstacle_polygons, 
                        point,
                        no_of_neighbors=no_of_neighbors,
                        safety_distance=safety_distance):
            valid_graph_node_locations.append(point)

    return valid_graph_node_locations


def a_star(graph, h, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:        
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:                
                    visited.add(next_node)               
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

    return path[::-1], path_cost


def extract_polygons(data):
    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        # Extract the 4 corners of the obstacle
        lower_left = north - d_north, east - d_east
        lower_right = north - d_north, east + d_east
        upper_left = north + d_north, east - d_east
        upper_right = north + d_north, east + d_east
        corners = [(lower_left), (lower_right), (upper_right), (upper_left)]

        height = alt + d_alt

        p = Polygon(corners)
        polygons.append((p, height))

    return polygons


def build_kd_tree_query(base_polygon_0, base_polygon_1, obj):
    """ 
    source: https://stackoverflow.com/questions/57273984/fast-way-to-find-the-closest-polygon-to-a-point/57274272?noredirect=1#comment101047819_57273984
    """
    dist_to_base_polygon_0 = base_polygon_0.distance(obj)
    dist_to_base_polygon_1 = base_polygon_1.distance(obj)
    return np.array([dist_to_base_polygon_0, dist_to_base_polygon_1])


def build_kd_tree(obstacle_polygons, use_poly_point_representation=True):
    if use_poly_point_representation:
        polygon_point_representation = get_2d_polygons_for_tree(obstacle_polygons)
        tree = KDTree(polygon_point_representation)
        return tree, None, None
    else:
        # define base polygons for k-d-tree
        base_polygon_0 = obstacle_polygons[0][0]
        base_polygon_1 = obstacle_polygons[1][0]

        distances = np.array([build_kd_tree_query(base_polygon_0, base_polygon_1, poly)
                              for (poly, h) in obstacle_polygons])
        tree = KDTree(distances)
        return tree, base_polygon_0, base_polygon_1


def get_2d_polygons_for_tree(polygons):
    polygons_tree_representation = []
    for (poly, h) in polygons:
        polygons_tree_representation.append([poly.representative_point().x, poly.representative_point().y])
    return polygons_tree_representation


def sample_2d_locations(data, num_of_samples, z_max):
    xmin = np.min(data[:, 0] - data[:, 3])
    xmax = np.max(data[:, 0] + data[:, 3])

    ymin = np.min(data[:, 1] - data[:, 4])
    ymax = np.max(data[:, 1] + data[:, 4])

    xvals = np.random.uniform(xmin, xmax, num_of_samples)
    yvals = np.random.uniform(ymin, ymax, num_of_samples)
    zvals = np.random.uniform(1.0, z_max, num_of_samples)

    return list(zip(xvals, yvals, zvals))


def collides(base_polygon_0, base_polygon_1, 
             tree, polygons, point, 
             no_of_neighbors, safety_distance):    
    if base_polygon_0 == None or base_polygon_1 == None:
        idx = tree.query([[point[0], point[1]]], k=no_of_neighbors, return_distance=False)[0]
    else:
        q = build_kd_tree_query(base_polygon_0, base_polygon_1, Point(point[0], point[1]))
        idx = tree.query([q], k=no_of_neighbors, return_distance=False)[0]

    for i in idx:
        nearest_poly = polygons[i]
        p = nearest_poly[0]
        h = nearest_poly[1]
        distance_to_border = p.distance(Point(point[0], point[1]))

        if ((p.contains(Point(point)) and (h >= point[2])) or (safety_distance > distance_to_border)):
            return True

    return False


def can_connect(polygons, p1, p2):
    point_1 = (p1[0], p1[1])
    point_2 = (p2[0], p2[1])
    line = [point_1, point_2]
    shapely_line = LineString(line)

    for (poly, h) in polygons:
        if (poly.crosses(shapely_line)) and h >= min(p1[2], p2[2]):
            return False
    return True


def create_graph(obstacle_polygons, valid_graph_node_locations, k):
    g = nx.Graph()
    tree = KDTree(valid_graph_node_locations)

    for n1 in valid_graph_node_locations:
        # for each node connect try to connect to k nearest nodes
        idxs = tree.query([n1], k, return_distance=False)[0]

        for idx in idxs:
            n2 = valid_graph_node_locations[idx]
            if n2 == n1:
                continue

            if can_connect(obstacle_polygons, n1, n2):
                g.add_edge(n1, n2, weight=1)
    return g


def get_closest_nodes(graph, current_point, no_of_nodes):
    """
    Find the k closest graph nodes to current_point.
    """
    node_distances = {}
    dist = 100000
    for p in graph.nodes:
        d = LA.norm(np.array(p) - np.array(current_point))
        node_distances[p] = d

    closest_nodes = sorted(node_distances, key=node_distances.get)
    closest_nodes = closest_nodes[0:(no_of_nodes - 1)]

    return closest_nodes


def get_closest_node(graph, point):
    """
    Find the closest graph node to certain point.
    """
    min_dist = 100000
    closest_node = (0, 0)
    for node in graph:
        dist_curr = distance.euclidean(node, point)
        if (dist_curr < min_dist):
            min_dist = dist_curr
            closest_node = node

    return closest_node


def find_collision_free_node(polygon_tree, no_of_neighbors, 
                             obstacle_polygons, potential_nodes, 
                             grid_point):
    """
    Finds the closest graph node to grid_point in the subset of potential_nodes 
    that can be reached without collision.
    """
    connection_found = False
    valid_node = None

    for node in potential_nodes:
        # get k nearest polygons to the grid point (node location could be chosen as well)
        idx_grid_point_closest_polygons = polygon_tree.query(
            [[grid_point[0], grid_point[1]]], k=no_of_neighbors, return_distance=False)[0]

        # check if the drone can fly from grid point to graph node without collision
        for i in idx_grid_point_closest_polygons:
            if (can_connect([obstacle_polygons[i]], grid_point, node)):
                connection_found = True
                valid_node = node
                break

            if connection_found:
                break
    return connection_found, valid_node


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def heuristic_handcrafted(position, goal_position):
    return math.sqrt(math.pow(position[0] - goal_position[0], 2) + math.pow(position[1] - goal_position[1], 2))


def heuristic_graph(n1, n2):
    return LA.norm(np.array(n2) - np.array(n1))
