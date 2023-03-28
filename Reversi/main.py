import math
import random
import sys

from game import Game   # 导入黑白棋文件
from copy import deepcopy
import time


class RandomPlayer:
    """
    随机玩家, 随机返回一个合法落子位置
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def random_choice(self, board):
        """
        从合法落子位置中随机选一个落子位置
        :param board: 棋盘
        :return: 随机合法落子位置, e.g. 'A1' 
        """
        # 用 list() 方法获取所有合法落子位置坐标列表
        action_list = list(board.get_legal_actions(self.color))

        # 如果 action_list 为空，则返回 None,否则从中选取一个随机元素，即合法落子坐标
        if len(action_list) == 0:
            return None
        else:
            return random.choice(action_list)

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        action = self.random_choice(board)
        return action


class HumanPlayer:
    """
    人类玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def get_move(self, board):
        """
        根据当前棋盘输入人类合法落子位置
        :param board: 棋盘
        :return: 人类下棋落子位置
        """
        # 如果 self.color 是黑棋 "X",则 player 是 "黑棋"，否则是 "白棋"
        if self.color == "X":
            player = "黑棋"
        else:
            player = "白棋"

        # 人类玩家输入落子位置，如果输入 'Q', 则返回 'Q'并结束比赛。
        # 如果人类玩家输入棋盘位置，e.g. 'A1'，
        # 首先判断输入是否正确，然后再判断是否符合黑白棋规则的落子位置
        while True:

            action = input(
                "请'{}-{}'方输入一个合法的坐标(e.g. 'D3'，若不想进行，请务必输入'Q'结束游戏。): ".format(player,
                                                                             self.color))

            # 如果人类玩家输入 Q 则表示想结束比赛
            if action == "Q" or action == 'q':
                return "Q"
            else:
                row, col = action[1].upper(), action[0].upper()

                # 检查人类输入是否正确
                if row in '12345678' and col in 'ABCDEFGH':
                    # 检查人类输入是否为符合规则的可落子位置
                    if action in board.get_legal_actions(self.color):
                        return action
                else:
                    print("你的输入不合法，请重新输入!")


class Node:
    def __init__(self, cur_board, parent=None, action=None, color=""):
        self.cur_board = cur_board  # 棋盘状态
        self.color = color  # 玩家颜色
        self.visits_cnt = 0  # 访问次数
        self.reward = 0.0  # 奖励期望
        self.parent = parent  # 父节点
        self.children = []  # 子节点
        self.action = action  # 行为

    def getUCB(self, ucb_param_C):
        if self.visits_cnt == 0:
            return sys.maxsize
        # 计算UCB
        explore = math.sqrt(
            2.0 * math.log(self.parent.visits_cnt) / float(self.visits_cnt))
        curUCB = self.reward/self.visits_cnt + ucb_param_C * explore
        return curUCB

    # 添加子节点
    def add_child(self, child_cur_board, action, color):
        child_node = Node(child_cur_board, parent=self,
                          action=action, color=color)
        self.children.append(child_node)

    # 是否已经完全扩展
    def is_fullexp(self):
        if len(self.children) == 0:
            return False
        for child in self.children:
            if child.visits_cnt == 0:
                return False
        return True


class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 'X' - 黑棋，'O' - 白棋
        """
        self.iter_times = 100  # 最大迭代次数
        self.ucb_param_C = 1  # ucb参数c
        self.max_step_count = 60  # 模拟过程中最多走多少步，60意味着走到游戏结束
        self.reward_bias = 10  # 胜负奖励偏移值
        self.color = color

    def MTCS(self, iter_times, root):
        """
        根据当前棋盘状态，使用MTCS算法，获取最佳落子位置
        :param iter_times: 最大迭代次数
        :param root: 根节点
        :return: action 落子位置
        """

        for i in range(iter_times):  # 迭代iter_times次
            node_select = self.select(root)
            node_leaf = self.extend(node_select)
            reward = self.simulate(node_leaf)
            self.backprop(node_leaf, reward)

        maxUCB = -sys.maxsize
        node_chose = None

        for child in root.children:
            childUCB = child.getUCB(self.ucb_param_C)
            if maxUCB < childUCB:
                maxUCB = childUCB
                node_chose = child

        return node_chose.action

    def select(self, node):
        """
        :param node:某节点
        :return: ucb值最大的叶节点
        """

        if len(node.children) == 0:
            return node
        if node.is_fullexp():    # 完全扩展
            max_node = None
            maxUCB = -sys.maxsize
            for child in node.children:
                childUCB = child.getUCB(self.ucb_param_C)
                if maxUCB < childUCB:
                    maxUCB = childUCB
                    max_node = child
            return self.select(max_node)
        else:   # 没有完全扩展
            for child in node.children:   # 从左向右遍历
                if child.visits_cnt == 0:
                    return child

    def extend(self, node):
        if node.visits_cnt == 0:    # 没被访问过，不扩展直接模拟
            return node
        else:   # 扩展
            next_color = 'X' if node.color == 'O' else 'O'
            for action in list(node.cur_board.get_legal_actions(node.color)):
                new_board = deepcopy(node.cur_board)
                new_board._move(action, node.color)
                # 创建新节点
                node.add_child(new_board, action=action, color=next_color)
            if len(node.children) == 0:
                return node
            return node.children[0]     # 返回第一个新孩子

    def simulate(self, node):
        """
        :param node:模拟起始点
        :return: 模拟结果reward
        """

        board = deepcopy(node.cur_board)
        step_count = 0
        color = node.color

        while (not self.game_over(board)) and step_count < self.max_step_count:   # 游戏没有结束，就模拟下棋
            action_list = list(board.get_legal_actions(color))

            if not len(action_list) == 0:   # 可以下，继续随机下棋
                action = random.choice(action_list)
                suc = board._move(action, color)
                color = 'X' if color == 'O' else 'O'
            else:   # 不能下，让对方下（此时必定可以下，否则游戏没结束的判断不会成立）
                color = 'X' if color == 'O' else 'O'
                action_list = list(board.get_legal_actions(color))
                action = random.choice(action_list)
                board._move(action, color)
                color = 'X' if color == 'O' else 'O'
            step_count += 1

        # winner:0-黑棋赢，1-白旗赢，2-表示平局
        # diff:赢家领先棋子数
        winner, diff = board.get_winner()
        reward = 0

        if winner == 2:
            reward = 0
        elif winner == 0:
            reward = diff + self.reward_bias
        else:
            reward = -(diff + self.reward_bias)
        if self.color == 'O':
            reward = - reward

        return reward

    def backprop(self, node, reward):
        """
        反向传播函数
        """
        while node is not None:
            node.visits_cnt += 1
            if node.color == self.color:
                node.reward -= reward
            else:
                node.reward += reward
            node = node.parent
        return 0

    def game_over(self, board):
        """
        判断游戏是否结束
        :return: True/False 游戏结束/游戏没有结束
        """
        # 根据当前棋盘，判断棋局是否终止
        # 如果当前选手没有合法下棋的位子，则切换选手；如果另外一个选手也没有合法的下棋位置，则比赛停止。
        b_list = list(board.get_legal_actions('X'))
        w_list = list(board.get_legal_actions('O'))

        is_over = (len(b_list) == 0 and len(w_list) == 0)  # 返回值 True/False

        return is_over

    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))

        time_begin = time.time()
        root = Node(cur_board=deepcopy(board), color=self.color)
        action = self.MTCS(self.iter_times, root)
        time_end = time.time()

        print(player_name + "思考完毕，该步用时{:.3f}秒".format(time_end - time_begin))
        print(player_name + "下在了" + action)
        return action


# AI黑棋初始化
ai_black_player = AIPlayer("X")

# AI白棋初始化
ai_white_player = AIPlayer("O")

# 随机玩家初始化
# random_black_player = RandomPlayer("X")
# random_white_player = RandomPlayer("O")

# 人类玩家初始化
# human_black_player = HumanPlayer("X")
# human_white_player = HumanPlayer("X")

# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(ai_black_player, ai_white_player)

# 开始下棋
game.run()
