import torch
import argparse
import numpy as np
import random

from env import Chess
from mcts import MCTSPlayer
from file_utils import *
from network import *
from collections import defaultdict, deque

parser = argparse.ArgumentParser()

""" Hyperparameter"""
parser.add_argument("--n_playout", type=int, default=5)

""" MCTS parameter """
parser.add_argument("--buffer_size", type=int, default=10000)
parser.add_argument("--c_puct", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
parser.add_argument("--self_play_sizes", type=int, default=100)
parser.add_argument("--training_iterations", type=int, default=440000)
parser.add_argument("--temp", type=float, default=1.0)

""" Policy update parameter """
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--learn_rate", type=float, default=1e-3)
parser.add_argument("--lr_mul", type=float, default=1.0)
parser.add_argument("--kl_targ", type=float, default=0.02)

""" Policy evaluate parameter """
parser.add_argument("--win_ratio", type=float, default=0.0)
parser.add_argument("--init_model", type=str, default=None)

args = parser.parse_args()

# make all args to variables
n_playout = args.n_playout
buffer_size = args.buffer_size
c_puct = args.c_puct
epochs = args.epochs
self_play_sizes = args.self_play_sizes
training_iterations = args.training_iterations
temp = args.temp
batch_size = args.batch_size
learn_rate = args.learn_rate
lr_mul = args.lr_mul
lr_multiplier = args.lr_multiplier
kl_targ = args.kl_targ
win_ratio = args.win_ratio
init_model = args.init_model


def collect_selfplay_data(env, mcts_player, game_iter):
    """collect self-play data for training"""
    data_buffer = deque(maxlen= 500 * 100)  # 400 (max len) * 50 (selfplay games)
    win_cnt = defaultdict(int)

    for self_play_i in range(self_play_sizes):
        rewards, play_data = self_play(env, mcts_player, temp, game_iter, self_play_i)
        play_data = list(play_data)[:]

        # augment the data
        data_buffer.extend(play_data)
        win_cnt[rewards] += 1

    print("\n ---------- Self-Play win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[0], win_cnt[-1]))

    win_ratio = 1.0 * win_cnt[1] / self_play_sizes
    print("Win rate : ", round(win_ratio * 100, 3), "%")
    wandb.log({"Win_Rate/self_play": round(win_ratio * 100, 3)})

    return data_buffer


def self_play(env, mcts_player, temp, game_iter=0, self_play_i=0):
    state = env.reset()
    player_0 = 0
    player_1 = 1 - player_0
    states, mcts_probs, current_player = [], [], []

    while True:
        available = []
        state = env.observe()
        available_actions = np.zeros(4672, )
        obs = torch.tensor(state.copy(), dtype=torch.float32)

        mask = env.legal_move_mask()
        indices = np.where(mask == 1.0)
        legal_move = list(zip(indices[0], indices[1], indices[2]))
        for move in legal_move:
            move_ = sensible_moves(env, move)
            available.append(uci_move_to_index(move_))

        for index in available:
            available_actions[index] = 1

        available_actions = torch.tensor(available_actions, dtype=torch.int8)

        # action_mask = torch.tensor(state['legal_moves'].copy(), dtype=torch.int8)
        combined_state = torch.cat([obs.flatten(), available_actions], dim=0)
        move, move_probs = mcts_player.get_action(env, game_iter, temp, return_prob=1)

        states.append(combined_state)
        mcts_probs.append(move_probs)
        current_player.append(player_0)

        env.step(move)

        player_0 = 1 - player_0
        player_1 = 1 - player_0

        if env.terminal is True:
            winner = -1
            wandb.log({"selfplay/reward": env.reward,
                       "selfplay/game_len": len(current_player)
                       })
            mcts_player.reset_player()  # reset MCTS root node

            if env.reward == -1:
                print("game: {}, self_play:{}, episode_len:{}".format(
                    game_iter + 1, self_play_i + 1, len(current_player)), "draw")
            else:
                print("game: {}, self_play:{}, episode_len:{}".format(
                    game_iter + 1, self_play_i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))

            if env.reward == 1:  # white win
                winner = 0
            elif env.reward == 0:  # black win
                winner = 1

            if env.reward != -1:  # env.reward :  +1 or -1 or -2
                winners_z[np.array(current_player) == winner] = 1.0  # win reward
                winners_z[np.array(current_player) != winner] = 0.0  # lose reward

            return env.reward, zip(states, mcts_probs, winners_z)


def policy_update(lr_mul, policy_value_net, data_buffers=None):
    """update the policy-value net"""
    kl, loss, entropy = 0, 0, 0
    lr_multiplier = lr_mul
    update_data_buffer = [data for buffer in data_buffers for data in buffer]

    mini_batch = random.sample(update_data_buffer, batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]

    state_batch_ = np.array([tensor.numpy() for tensor in state_batch])
    state_batch = state_batch_[:, :7616].reshape(128, 119, 8, 8)
    action_mask_batch = state_batch_[:, 7616:].reshape(128, 4672)
    old_probs, old_v = policy_value_net.policy_value(state_batch, action_mask_batch)

    for i in range(epochs):
        loss, entropy = policy_value_net.train_step(state_batch,
                                                    mcts_probs_batch,
                                                    winner_batch,
                                                    learn_rate * lr_multiplier)
        new_probs, new_v = policy_value_net.policy_value(state_batch, action_mask_batch)
        kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                            axis=1)
                     )
        if kl > kl_targ * 4:  # early stopping if D_KL diverges badly
            break
    # adaptively adjust the learning rate
    if kl > kl_targ * 2 and lr_multiplier > 0.1:
        lr_multiplier /= 1.5
    elif kl < kl_targ / 2 and lr_multiplier < 10:
        lr_multiplier *= 1.5

    print(("kl:{:.5f},"
           "lr_multiplier:{:.3f},"
           "loss:{},"
           "entropy:{}"
           ).format(kl, lr_multiplier, loss, entropy))

    return loss, entropy, lr_multiplier, policy_value_net


if __name__ == '__main__':
    # wandb intialize
    initialize_wandb(args, n_playout=n_playout)

    env = Chess()
    state = env.reset()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    if init_model:
        policy_value_net = PolicyValueNet(state.shape[0], state.shape[1], device, model_file=init_model)
    else:
        policy_value_net = PolicyValueNet(state.shape[0], state.shape[1], device)

    curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=1)
    data_buffer_training_iters = deque(maxlen=20)
    best_old_model, eval_model_file = None, None

    try:
        for i in range(training_iterations):
            """collect self-play data each iteration 1500 games"""
            data_buffer_each = collect_selfplay_data(env, curr_mcts_player, i)
            data_buffer_training_iters.append(data_buffer_each)

            """Policy update with data buffer"""
            loss, entropy, lr_multiplier, policy_value_net = policy_update(lr_mul=lr_multiplier,
                                                                           policy_value_net=policy_value_net,
                                                                           data_buffers=data_buffer_training_iters)
            wandb.log({"loss": loss,
                       "entropy": entropy})

            model_file, _ = create_models(n_playout, i)
            policy_value_net.save_model(model_file)

            if (i + 1) % 5 == 0:
                _, eval_model_file = create_models(n_playout, i)
                policy_value_net.save_model(eval_model_file)

    except KeyboardInterrupt:
        print('\n\rquit')