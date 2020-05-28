import sys
import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import pandas as pd
import utm
from sklearn.neighbors import KDTree
import pickle

import planning_utils as plu
import networkx as nx
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1],
                          self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self, waypoints):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(waypoints)
        self.connection._master.write(data)

    def is_goal_position_within_grid_boundaries(self, lon, lat, alt, grid_north_offset, grid_east_offset):
        proposed_global_goal_position = [lon, lat, alt]
        potential_local_position = global_to_local(proposed_global_goal_position, self.global_home)

        if (potential_local_position[0] < abs(grid_north_offset) and potential_local_position[0] > -abs(grid_north_offset)):
            if (potential_local_position[1] < abs(grid_east_offset) and potential_local_position[1] > -abs(grid_east_offset)):
                return True, potential_local_position
        return False, []

    def set_home_position_from_colliders(self, filepath):
        first_row = pd.read_csv(filepath, nrows=1)
        lon0 = float(first_row.columns[1].split()[1])
        lat0 = float(first_row.columns[0].split()[1])
        alt0 = 0.0
        self.set_home_position(lon0, lat0, alt0)

    def get_grid_start_and_goal_position(self, input_data, target_altitude, 
                                         safety_distance, 
                                         use_gedoetic_frame_for_goal_position=True):
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = plu.create_grid(input_data, target_altitude, safety_distance)

        # retrieve current global position
        current_global_position = [self._longitude, self._latitude, self._altitude]

        # convert to current local position relative to new global home based on colliders.csv using global_to_local()
        curr_local_position_rel_new_global_home = global_to_local(current_global_position, self.global_home)

        # Define starting point on the grid
        grid_start = (int(np.floor(curr_local_position_rel_new_global_home[0])), 
                      int(np.floor(curr_local_position_rel_new_global_home[1])),
                      curr_local_position_rel_new_global_home[2])

        if use_gedoetic_frame_for_goal_position:
            goal_position_valid = False
            count = 0
            max_proposals_allowed = 30
            while not goal_position_valid or count < max_proposals_allowed:
                goal_lon = np.random.uniform(-122.42, -122.3)
                goal_lat = np.random.uniform(37.7, 37.82)
                goal_position_valid, local_goal_position = self.is_goal_position_within_grid_boundaries(
                    goal_lon, goal_lat, 0.0, north_offset, east_offset)

                if (goal_position_valid):
                    grid_goal = (int(np.floor(local_goal_position[0])), 
                                 int(np.floor(local_goal_position[1])),
                                 -target_altitude)
                    break
                else:
                    count += 1
            if not goal_position_valid:
                print("no valid goal position found")
        else:
            # Set goal as some arbitrary position on the grid
            goal_north_position = np.random.randint(0, 2 * abs(north_offset))
            goal_east_position = np.random.randint(0, 2 * abs(east_offset))
            grid_goal = (goal_north_position, goal_east_position, -target_altitude)

        return grid, north_offset, east_offset, grid_start, grid_goal

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 3

        self.target_position[2] = TARGET_ALTITUDE

        # set home position to (lon0, lat0, 0) read from colliders.csv
        self.set_home_position_from_colliders("colliders.csv")

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # set up grid, start and goal position
        grid, north_offset, east_offset, grid_start, grid_goal = self.get_grid_start_and_goal_position(
            data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print('Local Start and Goal: ', grid_start, grid_goal)

        # -------------- commented out if obstacle polygons imported -------------
        # # extract polygons stating obstacles within the grid
        # print("Extracting polygons from obstacles...")
        # obstacle_polygons = plu.extract_polygons(data)
        # with open('pickled_obstacle_polygons.pickle', 'wb') as f:
        #     pickle.dump(obstacle_polygons, f)
        # ------------------------------------------------------------------------

        # -------------------- commented out if tree imported --------------------
        # # setup kdtree to later find closest polygons of colliding sample points
        # print("Setting up k-d-tree...")
        # tree, base_polygon_0, base_polygon_1 = plu.build_kd_tree(obstacle_polygons)
        # with open('pickled_tree.pickle', 'wb') as f:
        #     pickle.dump(tree, f)
        # ------------------------------------------------------------------------

        # ------------------- commented out if graph imported --------------------
        # # randomly select potential graph node locations within the grid
        # print("Finding graph node locations...")
        # np.random.seed(2)
        # random_graph_node_locations = plu.sample_2d_locations(data, 1000, 30)
        # t0 = time.time()
        # # extracting valid graph nodes keeping the given safety distance from the obstacles
        # valid_graph_node_locations = plu.get_valid_graph_node_locations(random_graph_node_locations, 
        #                                                                 base_polygon_0, 
        #                                                                 base_polygon_1, 
        #                                                                 tree, 
        #                                                                 obstacle_polygons,
        #                                                                 5,
        #                                                                 SAFETY_DISTANCE)
        # print('It took {0} seconds to find valid graph nodes'.format(time.time() - t0))
        # print("valid_graph_node_locations: ", len(valid_graph_node_locations))
        # t1 = time.time()
        # print("Creating graph...")
        # G = plu.create_graph(obstacle_polygons, valid_graph_node_locations, 10)
        # print('It took {0} minutes to build the graph'.format((time.time() - t1) / 60))
        # print("Number of edges", len(G.edges))

        # nx.write_gpickle(G, "pickled_graph_1000_random_nodes_30_zmax.pickle")
        # ------------------------------------------------------------------------

        # Read in obstacle polygons, tree and graph
        print("Reading in obstacle polygons...")
        with open("pickled_obstacle_polygons.pickle", 'rb') as f:
            obstacle_polygons = pickle.load(f)
        print("Reading in k-d-tree...")
        with open("pickled_tree.pickle", 'rb') as f:
            tree = pickle.load(f)
        print("Reading in graph...")
        graph_in = nx.read_gpickle("pickled_graph_1000_random_nodes_30_zmax.pickle")

        # set start and goal of graph
        print("Looking for a collision free start position...")
        graph_start_nodes = plu.get_closest_nodes(graph_in, grid_start, 10)
        start_connection_found, graph_start = plu.find_collision_free_node(
            tree, 10, obstacle_polygons, graph_start_nodes, grid_start)
        graph_goal = plu.get_closest_node(graph_in, grid_goal)

        if not start_connection_found:
            print("Cannot find collision free path to start location, please change position manually and try again.")
            self.landing_transition()
            self.disarming_transition()

        print("Planning path...")
        path, cost = plu.a_star(graph_in, plu.heuristic_graph, graph_start, graph_goal)
        # print(len(path), path)

        # Convert path to waypoints
        waypoints = [[p[0], p[1], p[2], 0] for p in path]

        # Set self.waypoints
        self.waypoints = waypoints

        # Send waypoints to sim (this is just for visualization of waypoints)
        waypoints_for_sim = [[int(p[0]), int(p[1]), int(p[2]), 0] for p in path]
        self.send_waypoints(waypoints_for_sim)

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
