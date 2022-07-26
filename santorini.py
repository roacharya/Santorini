import abc
import sys
import copy
import random

delta_x = {'n': -1, 'ne': -1, 'nw': -1, 'e': 0, 'w': 0, 's': 1, 'se': 1, 'sw': 1}
delta_y = {'n': 0, 'ne': 1, 'nw': -1, 'e': 1, 'w': -1, 's': 0, 'se': 1, 'sw': -1}
directions = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw']

class Board:
    def __init__(self, game, size = 5):
        self.size = size 
        self.row_str = "+--"*self.size + "+"
        self.heights = [[0]*self.size for _ in range(self.size)]
        self._game = game

    def get_height(self, xy_pos):
        return self.heights[xy_pos[0]][xy_pos[1]]

    def is_occupied(self, xy_pos):
        if self._game.p1.xy_pos1 == xy_pos or self._game.p1.xy_pos2 == xy_pos:
            return True
        if self._game.p2.xy_pos1 == xy_pos or self._game.p2.xy_pos2 == xy_pos:
            return True
        return False

    def __str__(self):
        xy_pos_dict = {tuple(self._game.p1.xy_pos1): 'A', tuple(self._game.p1.xy_pos2): 'B', tuple(self._game.p2.xy_pos1): 'Y', tuple(self._game.p2.xy_pos2): 'Z'}
        board_str = self.row_str + '\n'
        for i in range(self.size):
            board_str += '|'
            for j in range(self.size):
                player = ' '
                if (i,j) in xy_pos_dict:
                    player = xy_pos_dict[(i,j)]
                board_str += str(self.heights[i][j]) + player + '|'
            board_str += '\n' + self.row_str + '\n'
        return board_str

    @staticmethod
    def dist(xy_pos1, xy_pos2):
        return max(abs(xy_pos1[0] - xy_pos2[0]), abs(xy_pos1[1] - xy_pos2[1]))

class Move:
    def __init__(self, player_char, move_direction, build_direction):
            self.player_char = player_char
            self.move_direction = move_direction
            self.build_direction = build_direction

    def __str__(self):
            return "{}: {}, {}".format(self.player_char, self.move_direction, self.build_direction)

    @staticmethod
    def is_valid(board, xy_pos, direction, obj):
        new_x = xy_pos[0] + delta_x[direction]
        new_y = xy_pos[1] + delta_y[direction]

        if new_x < 0 or new_y < 0 or new_x >= len(board.heights) or new_y >= len(board.heights):
            return False
        if obj == 'move':
            if board.heights[new_x][new_y] > board.heights[xy_pos[0]][xy_pos[1]] + 1 or board.is_occupied([new_x, new_y]) or board.heights[new_x][new_y] > 3:
                return False
        elif obj == 'build':
            if board.is_occupied([new_x,new_y]) or board.heights[new_x][new_y] > 3:
                return False
        
        return True

class MoveIterable:
    def __init__(self, game, player):
        self.game = game
        self.player = player

    def __iter__(self):
        return MoveIterator(self.game, self.player)

class MoveIterator:
    def __init__(self, game, player):
        self.game = game
        self.player = player
        self.move_idx = 0
        self.build_idx = 0
        self.worker_idx = 0
        
    def __next__(self):
        if self.move_idx >= len(directions):
            self.build_idx += 1
            self.move_idx = 0
        if self.build_idx >= len(directions):
            self.worker_idx += 1
            self.build_idx = 0
        if self.worker_idx >= 2:
            raise StopIteration()

        xy_pos = self.player.xy_pos1.copy()
        if self.worker_idx == 1:
            xy_pos = self.player.xy_pos2.copy()
        
        move_direction = directions[self.move_idx]
        build_direction = directions[self.build_idx]
        if not Move.is_valid(self.game.board, xy_pos, move_direction, 'move'):
            self.move_idx += 1
            return self.__next__()

        xy_pos[0] += delta_x[move_direction]
        xy_pos[1] += delta_y[move_direction]

        if not Move.is_valid(self.game.board, xy_pos, build_direction, 'build'):
            self.move_idx += 1
            return self.__next__()

        self.move_idx += 1
        return Move(self.player.char_list[self.worker_idx], move_direction, build_direction)        

class BasePlayer(metaclass=abc.ABCMeta):
    def __init__(self, xy_pos1, xy_pos2, color, char_list):
        self.xy_pos1 = xy_pos1
        self.xy_pos2 = xy_pos2
        self.color = color
        self.char_list = char_list
        self.ch_pos_dict = {char_list[0]: self.xy_pos1, char_list[1]: self.xy_pos2}

    @abc.abstractmethod
    def getMove(self, game):
        pass

class HumanPlayer(BasePlayer):
    def getMove(self, game):
        char = input("Select a worker to move\n").strip()
        while char not in self.char_list:
            if char not in ['Y', 'Z', 'A', 'B']:
                char = input("Not a valid worker\nSelect a worker to move\n").strip()
            else:
                char = input("That is not your worker\nSelect a worker to move\n").strip()

        xy_pos = self.ch_pos_dict[char]
        move_direction = input("Select a direction to move (n, ne, e, se, s, sw, w, nw)\n")
        while True:
            if move_direction not in directions:
                print("Not a valid direction")
            elif not Move.is_valid(game.board, xy_pos, move_direction, 'move'):
                print("Cannot move {}".format(move_direction))
            else:
                break
            move_direction = input("Select a direction to move (n, ne, e, se, s, sw, w, nw)\n")
        
        xy_pos[0] += delta_x[move_direction]
        xy_pos[1] += delta_y[move_direction]
        build_direction = input("Select a direction to build (n, ne, e, se, s, sw, w, nw)\n")
        while True:
            if build_direction not in directions:
                print("Not a valid direction")
            elif not Move.is_valid(game.board, xy_pos, build_direction, 'build'):
                print("Cannot build {}".format(build_direction))
            else:
                break
            build_direction = input("Select a direction to build (n, ne, e, se, s, sw, w, nw)\n")        
        xy_pos[0] -= delta_x[move_direction]
        xy_pos[1] -= delta_y[move_direction]
        return Move(char, move_direction, build_direction)

class RandomPlayer(BasePlayer):
    def getMove(self, game):
        iter = MoveIterable(game, self)
        moves = [move for move in iter]
        move = random.choice(moves) 
        print("{},{},{}".format(move.player_char, move.move_direction, move.build_direction))
        return move

class HeuristicPlayer(BasePlayer):
    weights = (4,2.5,3)
    def getMove(self, game):
        print(game.board)
        iter = MoveIterable(game, self)
        best_score = -float("inf")
        best_move = None
        for move in iter:
            next_game = copy.deepcopy(game)
            if self is game.p1:
                next_player = next_game.p1
            else:
                next_player = next_game.p2
            next_game.make_move(next_player, move)
            heuristic = next_game.calc_score_tuple()
            new_score = sum(self.weights[i] * heuristic[i] for i in range(3))
            if new_score > best_score:
                best_score = new_score
                best_move = copy.deepcopy(move)
        print("{},{},{}".format(best_move.player_char, best_move.move_direction, best_move.build_direction))
        return best_move

class Memento:
    def __init__(self, game):
        self.board = copy.deepcopy(game.board)
        self.turn_num = game.turn_num
        self.p1 = copy.deepcopy(game.p1)
        self.p2 = copy.deepcopy(game.p2)

class Caretaker:
    def __init__(self, originator):
        self._mementos = []
        self._redo_list = []
        self._originator = originator

    def backup(self):
        self._mementos.append(self._originator.save())
    
    def undo(self):
        if not len(self._mementos):
            return

        memento = self._mementos.pop()
        cur = self._originator.save()
        self._redo_list.append(cur)
        self._originator.restore(memento)
    
    def redo(self):
        if not len(self._redo_list):
            return
        memento = self._redo_list.pop()
        cur = self._originator.save()
        self._mementos.append(cur)
        self._originator.restore(memento)
        return


class Observer(metaclass=abc.ABCMeta):
    def __init__(self, subject):
        self._subject = subject
    
    @abc.abstractmethod
    def update(self):
        pass

class BoardObserver(Observer):
    def update(self):
        print(self._subject.board, end='')

class HeuristicObserver(Observer):
    def update(self):
        color = ['blue', 'white'][self._subject.turn_num % 2]
        chars = ['YZ', 'AB'][self._subject.turn_num % 2]
        score = str(self._subject.calc_score_tuple())
        print("Turn: {}, {} ({}), {}".format(self._subject.turn_num, color, chars, score))

class InfoObserver(Observer):
    def update(self):
        color = ['blue', 'white'][self._subject.turn_num % 2]
        chars = ['YZ', 'AB'][self._subject.turn_num % 2]
        print("Turn: {}, {} ({})".format(self._subject.turn_num, color, chars))

class UndoObserver(Observer):
    def update(self):
        print("undo, redo, or next")

class Game:
    def __init__(self, p1, p2, undo, score):
        self.p1 = p1([3,1], [1,3], 'white', ['A', 'B'])
        self.p2 = p2([1,1], [3,3], 'blue', ['Y', 'Z'])
        self.board = Board(self)
        self.turn_num = 1
        self.observers = []
        self.observers.append(BoardObserver(self))
        if score:
            self.observers.append(HeuristicObserver(self))
        else:
            self.observers.append(InfoObserver(self))     
        if undo:
            self.observers.append(UndoObserver(self))
    
    def is_game_over(self):
        player, opponent = self.get_curr_players()
        
        if sum(1 for x in MoveIterable(self, player)) == 0:
            self.notify()
            print("{} has won".format(opponent.color))
            return True
        
        if self.board.get_height(player.xy_pos1) == 3 or self.board.get_height(player.xy_pos2) == 3:
            self.notify()
            print("{} has won".format(player.color))
            return True
        if self.board.get_height(opponent.xy_pos1) == 3 or self.board.get_height(opponent.xy_pos2) == 3:
            self.notify()
            print("{} has won".format(opponent.color))
            return True
        return False

    def get_curr_players(self):
        if self.turn_num % 2:
            return self.p1, self.p2
        return self.p2, self.p1

    def play_next_move(self):
        player, opponent = self.get_curr_players()
        move = player.getMove(self)
        self.make_move(player, move)
        self.turn_num += 1
    
    def make_move(self, player, move):
        xy_pos = player.ch_pos_dict[move.player_char]
        xy_pos[0] += delta_x[move.move_direction]
        xy_pos[1] += delta_y[move.move_direction]
        build_xy_pos = [xy_pos[0] + delta_x[move.build_direction], xy_pos[1] + delta_y[move.build_direction]]
        self.board.heights[build_xy_pos[0]][build_xy_pos[1]] += 1

    def notify(self):
        for observer in self.observers:
            observer.update()

    def save(self):
        return Memento(self)

    def restore(self, memento):
        self.p1 = copy.deepcopy(memento.p1)
        self.p2 = copy.deepcopy(memento.p2)
        self.turn_num = copy.deepcopy(memento.turn_num)
        self.board = copy.deepcopy(memento.board)
        self.board._game = self

    def calc_score_tuple(self):
        player, opponent = self.get_curr_players()

        height_score = self.board.get_height(player.xy_pos1) + self.board.get_height(player.xy_pos2)
        if self.board.get_height(player.xy_pos1) == 3 or self.board.get_height(player.xy_pos2) == 3:
            height_score = float("inf")

        inner_ring = [[1,1], [1,2],[1,3], [2,1], [2,3], [3,1], [3,2], [3,3]]
        center_score = int((player.xy_pos1 in inner_ring) + (player.xy_pos2 in inner_ring)+ (2*(player.xy_pos1 == [2,2])) + (2*(player.xy_pos2 == [2,2])))

        distance_score = min(Board.dist(player.xy_pos1, opponent.xy_pos1), Board.dist(player.xy_pos2, opponent.xy_pos1)) + \
            min (Board.dist(player.xy_pos1, opponent.xy_pos2), Board.dist(player.xy_pos2, opponent.xy_pos2))
        distance_score = 8 - distance_score
            
        return (height_score, center_score, distance_score)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        p1_type = sys.argv[1]
    else:
        p1_type = 'human'
    if len(sys.argv) >= 3:
        p2_type = sys.argv[2]
    else:
        p2_type = 'human'
    if len(sys.argv) >= 4:
        if sys.argv[3] == 'on':
            undo = True
        else:
            undo = False
    else:
        undo = False
    if len(sys.argv) >= 5:
        if sys.argv[4] == 'on':
            score = True
        else:
            score = False
    else:
        score = False
    player_class_mapping = {'human': HumanPlayer, 'random': RandomPlayer, 'heuristic': HeuristicPlayer}
    p1 = player_class_mapping[p1_type]
    p2 = player_class_mapping[p2_type]
    game = Game(p1, p2, undo, score)
    caretaker = Caretaker(game)
    while not game.is_game_over():
        game.notify()
        if undo:
            inp = input()
            while inp not in ['redo', 'undo', 'next']:
                inp = input("undo, redo, or next\n")
            if inp == 'undo':
                caretaker.undo()
            elif inp == 'redo':
                caretaker.redo()
            else:
                caretaker.backup()
                game.play_next_move()
                caretaker._redo_list = []
        else:
            game.play_next_move()
