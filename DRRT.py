import numpy as np
import matplotlib.pyplot as plt

class DRRT:
    def __init__(self, start, goal, obstacles, max_iter=1000, step_size=0.1):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = step_size
        self.tree = [start]
        self.path = []

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def is_collision(self, point):
        for obs in self.obstacles:
            if self.distance(point, obs['center']) <= obs['radius']:
                return True
        return False

    def get_random_point(self):
        return np.random.rand(2)

    def get_nearest_point(self, point):
        return min(self.tree, key=lambda p: self.distance(p, point))

    def steer(self, from_point, to_point):
        direction = np.array(to_point) - np.array(from_point)
        length = np.linalg.norm(direction)
        direction = direction / length
        return tuple(np.array(from_point) + direction * min(self.step_size, length))

    def replan(self):
        self.tree = [self.start]
        for _ in range(self.max_iter):
            rand_point = self.get_random_point()
            nearest_point = self.get_nearest_point(rand_point)
            new_point = self.steer(nearest_point, rand_point)
            if not self.is_collision(new_point):
                self.tree.append(new_point)
                if self.distance(new_point, self.goal) < self.step_size:
                    self.path = self.build_path(new_point)
                    break

    def build_path(self, end_point):
        path = [end_point]
        while path[-1] != self.start:
            nearest_point = self.get_nearest_point(path[-1])
            path.append(nearest_point)
        return path[::-1]

    def update_obstacles(self, new_obstacles):
        self.obstacles = new_obstacles
        self.replan()

    def plot(self):
        plt.figure()
        for obs in self.obstacles:
            circle = plt.Circle(obs['center'], obs['radius'], color='r')
            plt.gca().add_patch(circle)
        plt.plot(*zip(*self.tree), 'bo')
        if self.path:
            plt.plot(*zip(*self.path), 'g-')
        plt.plot(*self.start, 'go')
        plt.plot(*self.goal, 'ro')
        plt.show()

# Example usage
start = (0, 0)
goal = (1, 1)
obstacles = [{'center': (0.5, 0.5), 'radius': 0.1}]

drrt = DRRT(start, goal, obstacles)
drrt.replan()
drrt.plot()

# Update obstacles dynamically
new_obstacles = [{'center': (0.6, 0.6), 'radius': 0.1}]
drrt.update_obstacles(new_obstacles)
drrt.plot()
