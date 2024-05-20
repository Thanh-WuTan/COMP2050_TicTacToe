# Tic Tac Toe AI Project

This project includes my implementation of various AI algorithms that can play a classical game of Tic Tac Toe.

<p align="center">
  <img src="tictactoe.png" width=500/>
</p>

## Set up Environment
To set up the environment for this project, follow these steps: 
1. Install and setup [conda](https://docs.anaconda.com/free/miniconda/#quick-command-line-install).
2. Create environment.
```
conda create --name tictactoe python=3.8.19
conda activate tictactoe
pip install -r requirements.txt
```
## Project Structure
* [game.py](project/game.py) contains the game logic. 
* [gameplay.py](project/gameplay.py) contains game interactions between players (both AI and Human).
* [player.py](project/player.py) contains an abstract class for players.
* [tictactoe](project/tictactoe) folder contains AI agents for the game.

## Command-Line Usage

1. Quickstart
```
python main.py
```
2. Options
```
python main.py -p1 [PLAYER_1] -p2 [PLAYER_2] -m [VISUALIZATION] -n [NUM_GAMES] -t [TIMEOUT]
```
+ `--player1` or `-p1` : Choose player 1.
+ `--player1` or `-p1` : Choose player 2.
    + Choices of player: 'minimax', 'alphabeta', 'mcts', 'qlearning', 'human', 'random'.
+ `--mode` or `-m` : Choose visualization mode ('silent', 'plain', or 'ui'). 
    + 'silent' only shows game result (not possible for human player). 
    + 'plain' shows the game state in terminal. 
    + 'ui' shows the game with integrated UI.

+ `--num_games` or `-n` : Number of games for evaluations. Not that players will be assigned 'X' and 'O' alternately between games.
+ `--timeout` or `-t` : Set timeout for each AI move. No timeout is set for Human move. Default is 10 seconds per move.
+ `--no_timeout` or `-nt` :  No timeout for AI move.