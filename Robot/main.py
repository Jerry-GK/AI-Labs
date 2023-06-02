# 导入相关包
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot
import torch.nn.functional as F
from torch import optim
import torch
import numpy as np
from Maze import Maze
from Runner import Runner
from QRobot import QRobot
from ReplayDataSet import ReplayDataSet
from torch_py.MinDQNRobot import MinDQNRobot as TorchRobot  # PyTorch版本
from keras_py.MinDQNRobot import MinDQNRobot as KerasRobot  # Keras版本


def my_search(maze):
    class Node:
        def __init__(self, pos=(-1, -1)):
            self.pos = pos
            self.tail = None
            self.action = ''

    def move(pos, move):
        if move == 'u':
            return (pos[0] - 1, pos[1])
        elif move == 'r':
            return (pos[0], pos[1] + 1)
        elif move == 'd':
            return (pos[0] + 1, pos[1])
        elif move == 'l':
            return (pos[0], pos[1] - 1)

    def DFS_search(cur_pos):
        if cur_pos.pos == maze.destination:
            while cur_pos.tail is not None:
                path.append(cur_pos.action)
                cur_pos = cur_pos.tail
            path.reverse()
            return
        visit[cur_pos.pos] = True
        for m in maze.can_move_actions(cur_pos.pos):
            if not visit[move(cur_pos.pos, move=m)]:
                next = Node(pos=move(cur_pos.pos, move=m))
                next.action = m
                next.tail = cur_pos
                DFS_search(next)

    path = []
    visit = np.full(shape=(maze.maze_size, maze.maze_size), fill_value=False)
    start = Node(pos=maze.sense_robot())
    DFS_search(cur_pos=start)
    return path


class Robot(TorchRobot):
    actions = ['u', 'r', 'd', 'l']

    def __init__(self, maze):
        """
        初始化 Robot 类
        :param maze:迷宫对象
        """
        super(Robot, self).__init__(maze)
        maze.set_reward(reward={
            "hit_wall": 100,
            "destination": -1000,
            "default": 10
        }
        )
        self.maze = maze
        self.memory.build_full_view(maze=maze)
        self.losts = self.start_train()

    def start_train(self):
        losts = []
        batch_size = len(self.memory)
        size = size = self.maze.maze_size**2 - 1
        while True:
            loss = self._learn(batch_size)
            losts.append(loss)
            self.reset()
            for _ in range(size):
                _, reward = self.test_update()
                if reward == self.maze.reward["destination"]:
                    return losts

    def train_update(self):
        state = self.sense_state()
        action = self._choose_action(state)
        reward = self.maze.move_robot(action)
        return action, reward

    def test_update(self):
        state = torch.from_numpy(
            np.array(self.sense_state(), dtype=np.int16)).float().to(self.device)
        self.eval_model.eval()
        with torch.no_grad():
            q_val = self.eval_model(state).cpu().data.numpy()
        action = self.actions[np.argmin(q_val).item()]
        reward = self.maze.move_robot(action)
        return action, reward
