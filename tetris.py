# Tetris.py
import random
import json
from copy import deepcopy
from multiprocessing import Pool
import statistics
import sys
from tqdm import tqdm  # Optional, if you want a fancy progress bar

# Tetromino shape definitions
SHAPES = {
    'I': [[1,1,1,1]],
    'O': [[1,1],[1,1]],
    'T': [[0,1,0],[1,1,1]],
    'S': [[0,1,1],[1,1,0]],
    'Z': [[1,1,0],[0,1,1]],
    'J': [[1,0,0],[1,1,1]],
    'L': [[0,0,1],[1,1,1]]
}

def rotate(shape):
    return [list(row) for row in zip(*shape[::-1])]

# Precompute all rotations per piece
PIECES = {}
for name, shape in SHAPES.items():
    rots = []
    s = shape
    for _ in range(4):
        if s not in rots:
            rots.append(s)
        s = rotate(s)
    PIECES[name] = rots

# Assign each piece a fixed color value = (index+1)/7
PIECE_LIST = list(PIECES.keys())
PIECE_COLORS = {p: (i+1)/len(PIECE_LIST) for i, p in enumerate(PIECE_LIST)}

class Board:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0.0]*cols for _ in range(rows)]
        self.score = 0
        self.cleared_rows = []

    def valid_position(self, shape, x, y):
        for dy, row in enumerate(shape):
            for dx, cell in enumerate(row):
                if cell:
                    px, py = x+dx, y+dy
                    if px < 0 or px >= self.cols or py < 0 or py >= self.rows:
                        return False
                    if self.grid[py][px] != 0.0:
                        return False
        return True

    def place(self, shape, x, color):
        # drop piece
        y = 0
        while self.valid_position(shape, x, y+1):
            y += 1
        # lock piece with its color
        for dy, row in enumerate(shape):
            for dx, cell in enumerate(row):
                if cell:
                    self.grid[y+dy][x+dx] = color
        # clear full lines
        cleared = 0
        new_grid = []
        for row in self.grid:
            if all(val != 0.0 for val in row):
                self.cleared_rows.append(row.copy())
                cleared += 1
            else:
                new_grid.append(row)
        # pad at top
        for _ in range(cleared):
            new_grid.insert(0, [0.0]*self.cols)
        self.grid = new_grid
        self.score += cleared
        return cleared

def heuristic_score(grid, rows_cleared):
    rows = len(grid)
    cols = len(grid[0])
    # aggregate column heights
    agg = 0
    for x in range(cols):
        col = [grid[y][x] for y in range(rows)]
        height = rows - (next((i for i,v in enumerate(col) if v != 0.0), rows))
        agg += height
    # count holes
    holes = 0
    for x in range(cols):
        block = False
        for y in range(rows):
            if grid[y][x] != 0.0:
                block = True
            elif block:
                holes += 1
    # combine heuristics
    return -0.5*agg - 0.3*holes + 0.7*rows_cleared

def choose_move(board, piece, next_piece=None, use_lookahead=False):
    best = None
    best_score = float('-inf')
    color = PIECE_COLORS[piece]

    if not use_lookahead or next_piece is None:
        # No lookahead: normal move scoring
        for shape in PIECES[piece]:
            w = len(shape[0])
            for x in range(board.cols - w + 1):
                b = deepcopy(board)
                cleared = b.place(shape, x, color)
                s = heuristic_score(b.grid, cleared)
                if best and random.random() < 0.15:
                    continue
                if s > best_score:
                    best_score = s
                    best = (shape, x)
        return best

    # With lookahead: evaluate moves combined with next piece
    best_combo_score = float('-inf')
    for shape in PIECES[piece]:
        w = len(shape[0])
        for x in range(board.cols - w + 1):
            b = deepcopy(board)
            cleared = b.place(shape, x, color)
            # Now try all moves for next piece on this board
            next_color = PIECE_COLORS[next_piece]
            next_best_score = float('-inf')
            for next_shape in PIECES[next_piece]:
                nw = len(next_shape[0])
                for nx in range(b.cols - nw + 1):
                    b2 = deepcopy(b)
                    cleared2 = b2.place(next_shape, nx, next_color)
                    s2 = heuristic_score(b2.grid, cleared + cleared2)
                    if s2 > next_best_score:
                        next_best_score = s2
            # Combine current + best next move scores
            if next_best_score > best_combo_score:
                best_combo_score = next_best_score
                best = (shape, x)
    return best

def print_game_matrix(matrix, score):
    for row in matrix:
        print(''.join(['.' if cell == 0.0 else '#' for cell in row]))
    print(f'\nSCORE: {score}\n' + '-'*10 + '\n')

def simulate_one_game(args):
    rows, cols = args
    board = Board(rows, cols)
    bag = list(PIECES.keys())
    random.shuffle(bag)

    use_lookahead = random.random() < 0.05 #0.1  # 50% chance

    while True:
        if not bag:
            bag = list(PIECES.keys())
            random.shuffle(bag)
        piece = bag.pop()

        next_piece = None
        if use_lookahead:
            if not bag:
                temp_bag = list(PIECES.keys())
                random.shuffle(temp_bag)
                next_piece = temp_bag[0]
            else:
                next_piece = bag[-1]

        move = choose_move(board, piece, next_piece=next_piece, use_lookahead=use_lookahead)
        if not move or not board.valid_position(move[0], move[1], 0):
            break
        shape, x = move
        board.place(shape, x, PIECE_COLORS[piece])

    final_matrix = board.grid + board.cleared_rows
    # print_game_matrix(final_matrix, board.score)
    return {
        'game_matrix': final_matrix,
        'score': board.score
    }

import os

def build_tetris_data(num_games, rows, cols, output_file, mode="add"):
    data = []

    if mode == "add" and os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Warning: existing JSON file is invalid. Starting fresh.")
                data = []

    from tqdm import tqdm
    from multiprocessing import Pool

    with Pool(6) as pool:
        for result in tqdm(pool.imap_unordered(simulate_one_game, [(rows, cols)] * num_games), total=num_games):
            data.append(result)

    with open(output_file, 'w') as f:
        json.dump(data, f)

    scores = [d['score'] for d in data]
    print("\nFINAL STATS")
    print("Mean:", statistics.mean(scores))
    print("Min:", min(scores))
    print("Max:", max(scores))
    print("Std Dev:", statistics.stdev(scores))


#p 82/300 temp=1.00  train_corr=0.837  val_corr=0.861  train_corr_loss=0.1633  val_corr_loss=0.1394  actor_loss=0.0011  critic_loss=0.0013  design_var=0.0014  train_reward=0.0131