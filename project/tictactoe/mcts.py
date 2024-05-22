import numpy as np
import random
import math

from ..player import Player
from ..game import TicTacToe

from copy import deepcopy

WIN = 1
LOSE = -1
DRAW = 0
NUM_SIMULATIONS = 5000

class TreeNode():
    def __init__(self, game_state: TicTacToe, player_letter: str, parent=None, parent_action=None):
        self.player = player_letter
        self.game_state = game_state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.N = 0
        self.Q = 0 
    
    def select(self) -> 'TreeNode':
        """
        Select the best child node based on UCB1 formula. Keep selecting until a leaf node is reached.
        """
        if self.is_leaf_node():
            return self
        return self.best_child().select()
    
    def expand(self) -> 'TreeNode':
        """
        Expand the current node by adding all possible child nodes.
        """
        cur_player = self.game_state.curr_player         
        next_player = 'X' if cur_player == 'O' else 'O'   
        for cell in self.game_state.empty_cells():
            new_game_state = self.game_state.copy()
            new_game_state.set_move(cell[0], cell[1], cur_player)
            new_game_state.curr_player = next_player
            self.children.append(TreeNode(new_game_state, next_player, parent=self, parent_action=cell))
        rand_idx = random.randint(0, len(self.children) - 1)
        return self.children[rand_idx]
    
    def simulate(self) -> int:
        """
        Run simulation from the current node until the game is over. Return the result of the simulation.
        """ 
        game_state = self.game_state.copy()
        cur_player = game_state.curr_player
   
        while True:
            if game_state.game_over():
                if game_state.wins('X'):
                    return 'X'
                if game_state.wins('O'):
                    return 'O'
                return None
            random_move = random.choice(game_state.empty_cells())
            game_state.set_move(random_move[0], random_move[1], cur_player)
            if cur_player == 'X':
                cur_player = 'O'
            else:
                cur_player = 'X'
    def backpropagate(self, winner):
        """
        Backpropagate the result of the simulation to the root node.
        """
        self.N+= 1
        self.Q+= 1 if self.player != winner else 0
        if self.parent is not None:
            self.parent.backpropagate(winner)

    def is_leaf_node(self) -> bool:
        return len(self.children) == 0
    
    def is_terminal_node(self) -> bool:
        return self.game_state.game_over()
    
    def best_child(self) -> 'TreeNode':
        return max(self.children, key=lambda c: c.ucb())
    
    def ucb(self, c=math.sqrt(2)) -> float:
        if self.N == 0:
            return float('inf')
        return self.Q / self.N + c * np.sqrt(np.log(self.parent.N) / self.N)
    
class TTT_MCTSPlayer(Player):
    def __init__(self, letter, num_simulations=NUM_SIMULATIONS):
        super().__init__(letter)
        self.num_simulations = num_simulations
    
    def get_move(self, game):
        mcts = TreeNode(game, self.letter)
        
        for _ in range(self.num_simulations):
            leaf = mcts.select()
            chose_node = leaf
            if not leaf.is_terminal_node():
                chose_node = leaf.expand()
            winner = chose_node.simulate() 
            chose_node.backpropagate(winner) 
        
        best_child = max(mcts.children, key=lambda c: c.N)
        return best_child.parent_action
    
    def __str__(self) -> str:
        return "MCTS Player"