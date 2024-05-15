from SimNode import *
from utils import *

import os

from sklearn.neighbors import KDTree
from tqdm import tqdm

from heapq import heapify, heappush, heappop 

# Line ==> (x1,y1), (x2,y2)
# BBox ==> (min_x, min_y, min_z, max_x, max_y, max_z)
# Rectangle ==> (x, y, min_x, min_y, min_z, max_x, max_y, max_z, yaw)
# Point ==> (x, y)  


class PRM:
    def __init__(self, environment: Env, bot: Bot, n_points) -> None:
        self.environment = environment
        self.bot = bot
        
        self.env_rectangles = []
        self.env_lines = []

        for model_handle in self.environment.model_handles:
            x,y,z = self.environment.model_handles[model_handle]['position']
            _, _, yaw = self.environment.model_handles[model_handle]['orientation']
            bbox = self.environment.model_handles[model_handle]['bbox']
            points = getBBoxPoints(x, y, bbox, yaw)
            lines = getLines(points)
            self.env_lines = self.env_lines + lines
            self.env_rectangles.append([x,y] + bbox + [yaw])
        
        self.samples = []
        self.edges = {}
        self.tree = None
        self.goal = None

        self.a_star_e = 1.0

        self.n_points = n_points

    def checkEnvCollisionLine(self, line0):
        for line in self.env_lines:
            if checkCollisionBetweenLines(line, line0):
                return True
        return False
    
    def checkEnvCollisionRectangle(self, point, deflate = 0.0):
        for rectangle in self.env_rectangles:
            if checkPointInsideRectangle(point, rectangle, deflate = deflate):
                return True
        return False
    
    def find_neighbors(self, sample, k):
        distances, indices = self.tree.query([sample], k = k)
        return indices[0]
    
    def generate_samples(self, min_grid, max_grid):
        pbar = tqdm(total = self.n_points)
        while len(self.samples) < self.n_points:
            x,y = np.random.uniform([min_grid, min_grid], [max_grid, max_grid])
            if self.checkEnvCollisionRectangle((x,y)):
                continue
            self.samples.append([x,y])
            pbar.update()

    def add_edges(self, idx, neighbors_idx):
        for idx_n in neighbors_idx:
            if not self.checkEnvCollisionLine((self.samples[idx], self.samples[idx_n])):
                if idx in self.edges:
                    self.edges[idx].append(idx_n)
                else: 
                    self.edges[idx] = [idx_n]

    def build_roadmap(self, min_grid, max_grid, k):
        self.generate_samples(min_grid=min_grid, max_grid=max_grid)
        self.tree = KDTree(np.array(self.samples))
        
        for idx in tqdm(range(len(self.samples))):
            neighbors_idx = self.find_neighbors(self.samples[idx], k = k)
            self.add_edges(idx, neighbors_idx)
            
    def find_path(self, start, goal, deflate = 0.3):
        self.goal = goal
        if self.checkEnvCollisionRectangle(start, deflate=deflate):
            print('Start in Collision')
            return None
        if self.checkEnvCollisionRectangle(goal, deflate=deflate):
            print('Goal in Collision')
            return None

        start_neighbor_idx = None
        for neighbor_idx in sorted(list(range(len(self.samples))), key = lambda x: euclidean_distance(self.samples[x], start)):
            if not self.checkEnvCollisionLine([self.samples[neighbor_idx], start]) and not self.samples[neighbor_idx] == start:
                start_neighbor_idx = neighbor_idx
                break

        if start_neighbor_idx is None:
            print('No Start Neighbor Found')
            return None
        goal_neighbor_idx = None

        for neighbor_idx in sorted(list(range(len(self.samples))), key = lambda x: euclidean_distance(self.samples[x], goal)):
            if not self.checkEnvCollisionLine([self.samples[neighbor_idx], goal]) and not self.samples[neighbor_idx] == goal:
                goal_neighbor_idx = neighbor_idx
                break

        if goal_neighbor_idx is None:
            print('No Goal Neighbor Found')
            return None
        
        goal_node = None
        start_node = Node(self, start_neighbor_idx)
        fringe = [start_node]
        heapify(fringe)
        closed_set = set()

        while True:
            if len(fringe) == 0:
                print('Path Not Found')
                break
            
            curr_node: Node = heappop(fringe)
            closed_set.add(curr_node.idx)
            if curr_node.idx == goal_neighbor_idx:
                goal_node = curr_node
                break
            else:
                if not curr_node.idx in self.edges:
                    continue

                for idx_ in self.edges[curr_node.idx]:
                    if not idx_ in closed_set:
                        new_node = Node(self, idx_, curr_node)
                        if not idx_ in fringe:
                            heappush(fringe, new_node)
                        else:
                            idx_suc = fringe.index(idx_)
                            if fringe[idx_suc].g > curr_node.g + new_node.g:
                                fringe.pop(idx_suc)
                                heappush(fringe, new_node)
                            else:
                                del new_node
        return goal_node

    def plot_roadmap(self):
        for idx in self.edges:
            edge0 = self.samples[idx]
            for idx_ in self.edges[idx]:
                edge1 = self.samples[idx_]
                plt.plot([edge0[0], edge1[0]], [edge0[1], edge1[1]], 'g--')


class Node(object):
        def __init__(self, prm: PRM, idx, parent = None):
            self.idx = idx
            self.parent = parent
            x1,y1 = prm.samples[idx]
            x3, y3 = prm.goal
            self.h = (((x1 - x3) ** 2) + ((y1 - y3)**2)) ** 0.5
            self.e = prm.a_star_e
            if parent is not None:
                x2,y2 = prm.samples[parent.idx]
                self.g = (((x1 - x2) ** 2) + ((y1 - y2)**2)) ** 0.5
            else:
                self.g = 0
        
        def __repr__(self) -> str:
            if self.parent is not None:
                return f'{self.idx} << {self.parent.idx}'
            return f'{self.idx} << -1'

        def __eq__(self, __value: object) -> bool:
            return self.idx == __value
        
        def __lt__(self, other):
            return (self.g + self.e * self.h) < (other.g + self.e * other.h)
    
        
if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    models_path = home_dir + '/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu22_04/models/'
    obstacle_path = r'infrastructure/walls/240cm high walls/wall section 200cm.ttm'
    environment = Env(models_path)
    grid_size = 15

    # Spawn N obstacles
    for x in range(10):
        rnd = lambda : np.random.uniform(-grid_size, grid_size)
        rnd_theta = lambda : np.random.uniform(-np.pi, np.pi)
        environment.generate_object(obstacle_path, [rnd(), rnd(), 0], [0, 0, rnd_theta()], inflation = 0.6)

    bot = Bot()
    prm = PRM(environment, bot, n_points=1000)
    k = 20
    prm.build_roadmap(-grid_size, grid_size, k = k)
    # prm.plot_roadmap()
    # # for handle in environment.model_handles:
    # #         environment.plot_object(handle, style='b--')
    # plt.show()

    # while True: bot.moveArm(1.0)

    last_reached_wp = 0

    while True:
        # plt.clf()
        youbot_pose, youbot_pose_orientation = bot.getPoseState()
        dr12_pose, dr12_orientation = bot.getDR12State()
        g_dist = euclidean_distance(youbot_pose[:2] , dr12_pose[:2])
        if g_dist < 0.7:
            print('Planning Arm')
            if bot.moveArm([dr12_pose[0], dr12_pose[1], 0.4]):
                print('Catched!!')
                # environment.pause_env()

        start = youbot_pose[:2]
        goal = dr12_pose[:2]

        x_wp = [start[0], goal[0]]
        y_wp = [start[1] ,goal[1]]
        colors = ['r', 'k']
        # print(f'Finding Path from {start} {goal} {g_dist}')
        goal_node = prm.find_path(**{'start':start, 'goal':goal})
        if goal_node is None:
            print('NO FOUND')
            continue

        wp_idx_list = []
        while goal_node is not None:
            wp_idx_list.append(goal_node.idx)
            goal_node = goal_node.parent
        wp_idx_list = wp_idx_list[::-1]
        wp_list = [prm.samples[idx] for idx in wp_idx_list] + [goal]
        
        next_wp_idx = wp_idx_list[:2]
        if next_wp_idx == last_reached_wp:
            next_wp_idx = wp_idx_list[1:3]

        if len(next_wp_idx) > 1:
            reached = bot.lineFollow([prm.samples[idx] for idx in next_wp_idx], youbot_pose, youbot_pose_orientation, 0.5)
            if reached:
                last_reached_wp = next_wp_idx
                print('Reached!!')
        else:
            # bot.lineFollow([prm.samples[next_wp_idx[0]], goal], youbot_pose, youbot_pose_orientation, 0.1)
            # g_wp = prm.samples[next_wp_idx[0]]
            g_wp = goal
            g_yaw = np.arctan2(g_wp[1] - youbot_pose[1], g_wp[0] - youbot_pose[0])
            if bot.moveToGoal(g_wp + [g_yaw]):
                last_reached_wp = g_wp

        # x_values, y_values = zip(*wp_list)

        # plt.plot(x_values, y_values, marker = 'o', linestyle='-')
        # plt.scatter(x_wp, y_wp, c = colors, cmap='viridis')

        # for handle in environment.model_handles:
        #         environment.plot_object(handle, style='b--')
        # plt.pause(0.01)

plt.show()

