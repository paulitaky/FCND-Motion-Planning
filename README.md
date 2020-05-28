## Project: 3D Motion Planning
This project is a continuation of the Backyard Flyer project where a simple square shaped flight path was executed. In this project the taught techniques from the udacity lessons are used to plan a path through an urban environment.

![Quad Image](./misc/enroute.png)

---


# Required Steps for a Passing Submission:
1. Load the 2.5D map in the colliders.csv file describing the environment.
2. Discretize the environment into a grid or graph representation.
3. Define the start and goal locations.
4. Perform a search using A* or other search algorithm.
5. Use a collinearity test or ray tracing method (like Bresenham) to remove unnecessary waypoints.
6. Return waypoints in local ECEF coordinates (format for `self.all_waypoints` is [N, E, altitude, heading], where the drone’s start location corresponds to [0, 0, 0, 0].
7. Write it up.
8. Congratulations!  Your Done!

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it! Below I describe how I addressed each rubric point and where in my code each point is handled.

### Explain the Starter Code

#### 1. Explain the starter code provided in `motion_planning.py` and `planning_utils.py` compared to `backyard_flyer_solution.py`
Both scripts, in the following called backyard flyer and motion planner, implement the transitions between different flying states of the drone. While the backyard flyer calculates a simple box path without considering obstacles on the way, the motion planner starter code has an additional dedicated path planning state calculating the path from a certain start to goal location by running the a* algorithm. For this purpose a grid representation of the chosen city map below is set up considering grid cells occupied by obstacles. The planning utilities provide helper functions for setting up the grid, defining valid actions and an implementation of the grid based a* algorithm.

![Map of SF](./misc/map.png)

### Implementing Your Path Planning Algorithm

#### 1. Set your global home position
The global position setting is handled in an own function called *set_home_position_from_colliders(...)* covering the following steps:
- Read in the first line of the provided csv file with pandas
- Split the resulting strings and convert them to floats
- Set lat0, lon0 and alt0 by using the predefined function *self.set_home_position(...)*


#### 2. Set your current local position
Setting the drones current local position is part the function *self.get_grid_start_and_goal_position(...)* and is accomplished by executing the following steps:
- Set up grid representation considering safety margin around obstacles
- Use udacidrones predefined function global_to_local() to convert current global position to current local position with respect to previously set global home read in from the given csv
  

#### 3. Set grid start position from local position
The grid start location is set as well in the function *self.get_grid_start_and_goal_position(...)* by applying a floor and int casting operation to the previously calculated local position


#### 4. Set grid goal position from geodetic coords
As the third part of the function *self.get_grid_start_and_goal_position(...)* there are two possibilities implemented to set the goal position based on the grid:
- In case the geodetic frame is used, positions in geodetic coordinates are randomly sampled (uniform distribution) for *max_proposals_allowed* times and checked if they lie within the grid boundaries by the function called *self.is_goal_position_within_grid_boundaries(...)*. The goal location is set to the first found valid location in geodetic coordinates.
- In case the geodetic frame is not used, the goal position is set to a random location on the grid by randomly choosing grid cell indices.

#### 5. Modify A* to include diagonal motion (or replace A* altogether)
In this project implementation the **probabilistic roadmap approach** is used, applying the graph based a* algorithm to plan a collision free path from a start to a goal (graph node) location. As it takes quite some time to create the graph, it was created once, stored in a pickle file and read in for following runs. The same procedure was applied to the used polygons representing the obstacles and the k-d-tree. The more detailed steps are as follows (*plu == planning_utilities.py*):
- Extract polygons stating obstacles within the grid -> *plu.extract_polygons(data)*
- Setup k-d-tree to later find closest polygons of colliding sample points. Hereby, the k-d-tree is calculated based on the obstacle polygons described by one representative point -> *plu.build_kd_tree(obstacle_polygons)* 
- Randomly select potential graph node locations within the grid -> *plu.sample_2d_locations(data, 1000, 20)*
- Extract valid graph nodes, i.e. not colliding with obstacles, while keeping the given safety distance -> *plu.get_valid_graph_node_locations(random_graph_node_locations, base_polygon_0, base_polygon_1, tree, obstacle_polygons, 5, SAFETY_DISTANCE)*
- Create graph out of edges connecting valid nodes without collision -> *plu.create_graph(obstacle_polygons, valid_graph_node_locations, 10)*
- Set graph start position based on a collision free connection between grid start point and closest graph node -> *plu.get_closest_nodes(graph_in, grid_start, 10)* and *plu.find_collision_free_node(tree, 10, obstacle_polygons, graph_start_nodes, grid_start)*
- Set graph goal location based on closest graph node -> *plu.get_closest_node(graph_in, grid_goal)*
- Apply a* algorithm based on graph nodes -> *plu.a_star(graph_in, plu.heuristic_graph, graph_start, graph_goal)*
- Extract and set global waypoints from found path
- Map waypoints to integer values for simulator waypoint visualization and send waypoints to simulator
  

#### 6. Cull waypoints 
Not implemented as probabilistic roadmap approach is used.



### Execute the flight
#### 1. Does it work?
It works!

### Double check that you've met specifications for each of the [rubric](https://review.udacity.com/#!/rubrics/1534/view) points.
  
# Extra Challenges: Real World Planning

For an extra challenge, consider implementing some of the techniques described in the "Real World Planning" lesson. You could try implementing a vehicle model to take dynamic constraints into account, or implement a replanning method to invoke if you get off course or encounter unexpected obstacles.


