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
        while True:
            if self.is_leaf_node():
                return self
            children = self.children
            max_ucb = -math.inf
            for child in children:
                if child.ucb() > max_ucb:
                    max_ucb = child.ucb()
                    self = child       
    
    def expand(self) -> 'TreeNode':
        """
        Expand the current node by adding all possible child nodes. Return one of the child nodes for simulation.
        """
        empty_cells = self.game_state.empty_cells()
        next_player = self.game_state.curr_player
        for empty_cell in empty_cells:
            x, y = empty_cell[0], empty_cell[1]
            new_game_state = deepcopy(self.game_state)
            new_game_state.curr_player = next_player 
            new_game_state.set_move(x, y, next_player)
            child = TreeNode(game_state=new_game_state, player_letter=next_player, parent=self, parent_action=[x, y])
            self.children.append(child)

        rand_idx = random.randint(0, len(self.children) - 1)
        return self.children[rand_idx]
    
    def simulate(self) -> int:
        """
        Run simulation from the current node until the game is over. Return the result of the simulation.
        """
        game_state = self.game_state.copy()
        player = deepcopy(self.player)
        while True:
            if game_state.game_over():
                if game_state.wins('X'):
                    return 'X'
                if game_state.wins('O'):
                    return 'O'
                return None
            if player == 'X':
                player = 'O'
            else:
                player = 'X'
            empty_cells = game_state.empty_cells()
            rand_idx = random.randint(0, len(empty_cells) - 1)
            action = empty_cells[rand_idx]
            x, y = action[0], action[1]
            game_state.set_move(x, y, player)
            
    def backpropagate(self, winner: int):
        """
        Backpropagate the result of the simulation to the root node.
        """
        cur_node = self
        while True:
            cur_node.N+= 1
            if winner != None:
                if cur_node.player == winner:
                    cur_node.Q+= 1
            if cur_node.parent == None:
                break
            else:
                cur_node = cur_node.parent
    

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
            if not leaf.is_terminal_node():
                chose_node = leaf.expand()
            else:
                chose_node = leaf
            winner = chose_node.simulate() 
            chose_node.backpropagate(winner) 
     
        best_child = max(mcts.children, key=lambda c: c.N)
        return best_child.parent_action
    
    def __str__(self) -> str:
        return "MCTS Player"