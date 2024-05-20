import random
import math
from ..player import Player
from ..game import TicTacToe

class TTT_AlphaBetaPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game: TicTacToe):
        depth = len(game.empty_cells())
        if depth == 0 or game.game_over():
            return
        
        if len(game.empty_cells()) == 9:
            move = random.choice(game.empty_cells())
        else:
            # Alpha-Beta Pruning: Initialize alpha to negative infinity and beta to positive infinity
            alpha = -math.inf
            beta = math.inf
            choice = self.minimax(game, depth, self.letter, alpha, beta)
            move = [choice[0], choice[1]]
        return move

    def minimax(self, game: TicTacToe, depth: int, player_letter: str, alpha: float, beta: float):
        """
        AI function that chooses the best move with alpha-beta pruning.
        :param game: current game state
        :param depth: node index in the tree (0 <= depth <= 9)
        :param player_letter: value representing the player
        :param alpha: best value that the maximizer can guarantee
        :param beta: best value that the minimizer can guarantee
        :return: (row, col) of the selected move
        """ 
        if depth == 0 or game.game_over():
            return [None, None, self.evaluate(game)]
        best = [None, None]
        for empty_cell in game.empty_cells(state=game.board_state):
            x, y = empty_cell[0], empty_cell[1]
            newgame = game.copy()
            newgame.curr_player = player_letter
            newgame.set_move(x, y, player_letter)
            new_player_letter = 'X'
            if player_letter == 'X':
                new_player_letter = 'O'
            score = self.minimax(newgame, depth - 1, new_player_letter, alpha, beta)
            score[0], score[1] = x, y
            if player_letter == 'X':
                if score[2] > alpha:
                    alpha = score[2]
                    best = [x, y]
            else:
                if score[2] < beta:
                    beta = score[2]
                    best = [x, y]
        if player_letter == 'X':
            return [best[0], best[1], alpha]
        else:
            return [best[0], best[1], beta]
    
    def evaluate(self, game: TicTacToe) -> int:
        """
        Function to evaluate the score of game state.
        :param game: the game state to evaluate
        :return: the score of the board from the perspective of current player
        """
        score = 0
        if game.wins('X'):
            score = 1
        elif game.wins('O'):
            score = -1
        else:
            score = 0
        return score
    
    def __str__(self) -> str:
        return "Alpha-Beta Player"