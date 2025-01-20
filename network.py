import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import os
import chess


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_masked_act_probs(log_act_probs, action_mask_batch):
    act_probs = torch.exp(log_act_probs).cpu().numpy()
    masked_act_probs = act_probs * action_mask_batch

    if action_mask_batch.sum(axis=1).all() > 0:
        act_probs = masked_act_probs / masked_act_probs.sum(axis=1, keepdims=True)
    else:
        act_probs = masked_act_probs / (masked_act_probs.sum() + 1)

    return act_probs


def uci_move_to_index(move):
    from_square = move.from_square
    to_square = move.to_square

    move_index = 64 * from_square + to_square

    return move_index


def index_to_uci_move(index):
    from_square = index // 64
    to_square = index % 64
    move = chess.Move(from_square, to_square)

    return move, from_square, to_square


def sensible_moves(env, legal_move):
    fromRow, fromCol, plane = legal_move

    if plane < 56:  # Queen move
        squares, direction = divmod(plane, 8)
        squares += 1

        """
        7 0 1
        6   2
        5 4 3
        """
        if direction == 0:
            toRow = fromRow - squares
            toCol = fromCol
        elif direction == 1:
            toRow = fromRow - squares
            toCol = fromCol + squares
        elif direction == 2:
            toRow = fromRow
            toCol = fromCol + squares
        elif direction == 3:
            toRow = fromRow + squares
            toCol = fromCol + squares
        elif direction == 4:
            toRow = fromRow + squares
            toCol = fromCol
        elif direction == 5:
            toRow = fromRow + squares
            toCol = fromCol - squares
        elif direction == 6:
            toRow = fromRow
            toCol = fromCol - squares
        else:  # direction == 7
            toRow = fromRow - squares
            toCol = fromCol - squares

        fromSquare = (7 - fromRow) * 8 + fromCol
        toSquare = (7 - toRow) * 8 + toCol
        move = chess.Move(fromSquare, toSquare)
    elif plane < 64:  # Knight move
        """
        [ ][5][ ][3][ ]
        [7][ ][ ][ ][1]
        [ ][ ][K][ ][ ]
        [6][ ][ ][ ][0]
        [ ][4][ ][2][ ]
        """
        if plane == 56:
            toRow = fromRow + 1
            toCol = fromCol + 2
        elif plane == 57:
            toRow = fromRow - 1
            toCol = fromCol + 2
        elif plane == 58:
            toRow = fromRow + 2
            toCol = fromCol + 1
        elif plane == 59:
            toRow = fromRow - 2
            toCol = fromCol + 1
        elif plane == 60:
            toRow = fromRow + 2
            toCol = fromCol - 1
        elif plane == 61:
            toRow = fromRow - 2
            toCol = fromCol - 1
        elif plane == 62:
            toRow = fromRow + 1
            toCol = fromCol - 2
        else:  # plane == 63
            toRow = fromRow - 1
            toCol = fromCol - 2

        fromSquare = (7 - fromRow) * 8 + fromCol
        toSquare = (7 - toRow) * 8 + toCol
        move = chess.Move(fromSquare, toSquare)
    else:  # Underpromotions
        toRow = fromRow - env.board.turn

        if plane <= 66:
            toCol = fromCol
            promotion = plane - 62
        elif plane <= 69:
            diagonal = 0
            promotion = plane - 65
            toCol = fromCol - env.board.turn
        else:  # plane <= 72
            diagonal = 1
            promotion = plane - 68
            toCol = fromCol + env.board.turn

        fromSquare = (7 - fromRow) * 8 + fromCol
        toSquare = (7 - toRow) * 8 + toCol
        move = chess.Move(fromSquare, toSquare, promotion=promotion)

    return move


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        self.conv1 = nn.Conv2d(119, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height, board_width * board_height * 73)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 256)
        self.val_fc2 = nn.Linear(256, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height, device, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        self.device = device

        # the policy value net module
        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)
        if model_file:
            net_params = torch.load(model_file, map_location=self.device, weights_only=True)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch, action_mask_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = compute_masked_act_probs(log_act_probs, action_mask_batch)

        return act_probs, value

    def policy_value_fn(self, env):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        available = []
        mask = env.legal_move_mask()
        indices = np.where(mask == 1.0)
        legal_move = list(zip(indices[0], indices[1], indices[2]))

        for move in legal_move:
            move_ = sensible_moves(env, move)
            available.append(uci_move_to_index(move_))

        current_state = env.observe()
        current_state = torch.tensor(current_state.copy(), dtype=torch.float32)
        current_state = current_state.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (8, 8, 119) -> (1, 119, 8, 8)

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(current_state)
            act_probs = torch.exp(log_act_probs).cpu().numpy().flatten()
            masked_act_probs = np.zeros_like(act_probs)
            masked_act_probs[available] = act_probs[available]

            if masked_act_probs.sum() > 0:  # if have not available action
                masked_act_probs /= masked_act_probs.sum()
            else:
                masked_act_probs /= (masked_act_probs.sum()+1)

        return available, masked_act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        state_batch_np = np.array(state_batch)
        mcts_probs_np = np.array(mcts_probs)
        winner_batch_np = np.array(winner_batch)

        # numpy array to tensor
        state_batch = torch.tensor(state_batch_np, dtype=torch.float32, device=self.device)
        mcts_probs = torch.tensor(mcts_probs_np, dtype=torch.float32, device=self.device)
        winner_batch = torch.tensor(winner_batch_np, dtype=torch.float32, device=self.device)

        log_act_probs, value = self.policy_value_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        set_learning_rate(self.optimizer, lr)

        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        # Ensure that the directory exists before saving the file
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        torch.save(net_params, model_file)