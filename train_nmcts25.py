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
parser.add_argument("--n_playout", type=int, default=25)

""" MCTS parameter """
parser.add_argument("--buffer_size", type=int, default=10000)
parser.add_argument("--c_puct", type=int, default=5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr_multiplier", type=float, default=1.0)
parser.add_argument("--self_play_sizes", type=int, default=100)
parser.add_argument("--training_iterations", type=int, default=2000)
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
    data_buffer = deque(maxlen= 400 * 50 * 8)  # 400 (max len) * 50 (selfplay games) * 8 (equi)
    win_cnt = defaultdict(int)

    for self_play_i in range(self_play_sizes):
        rewards, play_data = self_play(env, mcts_player, temp, game_iter, self_play_i)
        play_data = list(play_data)[:]

        # augment the data
        # play_data = get_equi_data(play_data)
        data_buffer.extend(play_data)
        win_cnt[rewards] += 1

    print("\n ---------- Self-Play win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[-1], win_cnt[0]))

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
            wandb.log({"selfplay/reward": env.reward,
                       "selfplay/game_len": len(current_player)
                       })

            if env.reward == 0:
                print('self_play_draw')

            mcts_player.reset_player()  # reset MCTS root node
            print("game: {}, self_play:{}, episode_len:{}".format(
                game_iter + 1, self_play_i + 1, len(current_player)))
            winners_z = np.zeros(len(current_player))

            if env.reward != 0:  # non draw
                if env.reward == -1:
                    env.reward = 0
                # if winner is current player, winner_z = 1
                winners_z[np.array(current_player) == 1 - env.reward] = 1.0
                winners_z[np.array(current_player) != 1 - env.reward] = -1.0
                if env.reward == 0:
                    env.reward = -1
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
    state_batch = state_batch_[:, :7616].reshape(64, 119, 8, 8)
    action_mask_batch = state_batch_[:, 7616:].reshape(64, 4672)
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


def policy_evaluate(env, current_mcts_player, old_mcts_player, n_games=30):  # total 30 games
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    training_mcts_player = current_mcts_player  # training Agent
    opponent_mcts_player = old_mcts_player
    win_cnt = defaultdict(int)

    for j in range(n_games):
        winner, move_len = start_play(env, training_mcts_player, opponent_mcts_player)
        win_cnt[winner] += 1
        print("{} / 30 - move_len: {} ".format(j + 1, move_len))

    win_ratio = 1.0 * win_cnt[1] / n_games
    print("---------- win: {}, lose: {}, tie:{} ----------".format(win_cnt[1], win_cnt[-1], win_cnt[0]))
    return win_ratio, training_mcts_player


def start_play(env, player1, player2):
    """start a game between two players"""
    state = env.reset()
    players = [0, 1]
    p1, p2 = players
    player1.set_player_ind(p1)
    player2.set_player_ind(p2)
    players = {p1: player1, p2: player2}
    current_player = 0
    player_in_turn = players[current_player]
    move_list = []

    while True:
        # synchronize the MCTS tree with the current state of the game
        move = player_in_turn.get_action(env, game_iter=-1, temp=1e-3, return_prob=0)
        move_list.append(move)
        env.step(move)

        if not env.terminal:
            current_player = 1 - current_player
            player_in_turn = players[current_player]

        else:
            wandb.log({"eval/game_len": len(move_list),
                       "eval/reward": env.reward
                       })
            return env.reward, len(move_list)


if __name__ == '__main__':
    # wandb intialize
    initialize_wandb(args, n_playout=n_playout)

    env = Chess()
    state = env.reset()

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda:0")

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
            if i == 0:
                policy_evaluate(env, curr_mcts_player, curr_mcts_player)
                model_file, eval_model_file = create_models(n_playout, i)
                policy_value_net.save_model(model_file)
                policy_value_net.save_model(eval_model_file)

            else:
                existing_files = get_existing_files(n_playout=n_playout)
                old_i = max(existing_files)
                best_old_model, _ = create_models(n_playout, (old_i - 1))
                policy_value_net_old = PolicyValueNet(state.shape[0], state.shape[1], device, best_old_model)

                old_mcts_player = MCTSPlayer(policy_value_net_old.policy_value_fn, c_puct, n_playout, is_selfplay=0)
                curr_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct, n_playout, is_selfplay=0)

                win_ratio, curr_mcts_player = policy_evaluate(env, curr_mcts_player, old_mcts_player)

                if (i + 1) % 10 == 0:  # save model 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 (1+10: total 11)
                    _, eval_model_file = create_models(n_playout, i)
                    policy_value_net.save_model(eval_model_file)

                print("Win rate : ", round(win_ratio * 100, 3), "%")
                wandb.log({"Win_Rate/Evaluate": round(win_ratio * 100, 3)})

                if win_ratio > 0.5:
                    old_mcts_player = curr_mcts_player
                    model_file, _ = create_models(n_playout, i)
                    policy_value_net.save_model(model_file)
                    print(" ---------- New best policy!!! ---------- ")

                else:
                    # if worse it just reject and does not go back to the old policy
                    print(" ---------- Low win-rate ---------- ")

    except KeyboardInterrupt:
        print('\n\rquit')