In this problem we are given a robot operating
in a 2D grid world. Every cell in the grid world is characterized by a color (0 or
1). The robot is equipped with a noisy odometer and a noisy color sensor. Given
a stream of actions and corresponding observations, implement a Bayes filter to
keep track of the robot’s current position. The sensor reads the color of cell of the
grid world correctly with probability 0.9 and incorrectly with probability 0.1. At
each step, the robot can take an action to move in 4 directions (north, east, south,
west). Execution of these actions is noisy, so after the robot performs this action,
it actually makes the move with probability 0.9 and stays at the same spot without
moving with probability 0.1.
When the robot is at the edge of the grid world and is tasked with executing an
action that would take it outside the boundaries of the grid world, the robot remains
in the same state with probability 
