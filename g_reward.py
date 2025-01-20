import jax
import pickle
import numpy as np
import os
import jax.numpy as jnp
from collections import Counter


os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'platform'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"

def initialize_model(path):
    model_path = os.path.join(path, "model.pkl")

    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt['reward_model']
    return reward_model

def encode_action(action, num_classes=4672):
    action = action - 1
    one_hot_action = np.eye(num_classes)[action]
    return one_hot_action

def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)

def reward_from_preference_transformer(observation, action,reward_model, len_query=50 ):

    # for online learning skip this code and input observation/action data
    # observation data shape = (game_len, 8, 8, 119)
    # action data shape = (game_len, )
    # And in this code, the action is converted to one-hot encoded action
    # example of input action style  : 7472
    # for saved data
    # observation = observation.squeeze(axis=0)
    # action = action.squeeze(axis=0)
    observations = []
    actions = []
    timesteps = []
    attn_masks = []

    observation_ = observation
    actions_ = action
    timesteps_ = [i for i in range(len(action))]
    attn_mask_ = np.zeros(len(action), dtype=np.int32)
    attn_mask_[:observation_.shape[0]] = 1

    width = len_query - observation_.shape[0]

    if width > 0:
        observation_ = np.pad(observation_, ((0, width), (0, 0), (0, 0), (0, 0)), mode='constant')
        actions_ = np.pad(actions_, (0, width), mode='constant')
        timesteps_ = np.pad(timesteps_, (0, width), mode='constant')
        attn_mask_ = np.pad(attn_mask_, (0, width), mode='constant')


    # attn_mask_ = np.zeros(len(action), dtype=np.int32)
    # attn_mask_[:observation_.shape[0] ] = 1

    max_idx = max(0, len(observation_) - len_query)
    idxx = int(np.random.choice(np.linspace(0, max_idx, max_idx + 1)))

    observations.append(observation_[idxx:idxx + len_query])
    actions.append(actions_[idxx:idxx + len_query])
    timesteps.append(timesteps_[idxx:idxx + len_query])
    attn_masks.append(attn_mask_[idxx:idxx + len_query])

    input = {
        "observations": np.array(observations, dtype=float),
        "actions": np.array(actions),
        "timestep": np.array(timesteps),
        "attn_mask": np.array(attn_masks)
    }

    input['observations'] = input['observations'].reshape(1, len_query, 7616)
    input['actions'] = encode_action(input['actions'], 4672)

    jax_input = batch_to_jax(input)
    new_reward, w = reward_model.get_reward(jax_input)
    new_reward = new_reward.reshape(1, len_query) * input['attn_mask']
    new_reward = jnp.sum(new_reward, axis=1) / jnp.sum(input['attn_mask'], axis=1)
    new_reward = new_reward.reshape(-1, 1)
    new_reward = np.asarray(list(new_reward))
    new_reward = new_reward.squeeze(-1)
    new_reward = new_reward[0]

    return new_reward

def compare_moves(actions):
    open_moves = {'A10': [666, 3364, 918, 3299, 1699, 4013, 334, 2915, 405],
                  'D06': [731, 3299, 666, 3372, 405, 4013, 166, 3567],
                  'A46': [731, 4013, 405, 3372, 666, 3956, 82, 262, 788],
                  'E00': [731, 4013, 666, 3372, 405, 3299, 75, 3567],
                  'E61': [731, 4013, 666, 3502, 82, 3958, 796, 3307, 405],
                  'C00': [796, 3372, 731, 3299, 82, 3929, 1828, 3234],
                  'B50': [796, 3234, 405, 3307, 731, 2203, 1371, 4013, 82],
                  'B30': [796, 3234, 82, 3690, 405, 3372, 731, 2203, 1371],
                  'B40': [796, 3234, 405, 3372, 731, 2203, 1371, 4013, 82],
                  'C60': [796, 3364, 405, 3690, 353, 3112, 2136, 4013, 262],
                  'B10': [796, 3242, 731, 3299, 75, 2268, 732, 4013, 1837],
                  'A05': [405, 4013, 918, 3502, 334, 3958, 262, 3902, 731]}

    for opening, moves in open_moves.items():
        actions = actions[:len(moves)]
        if actions == moves:
            mov = opening
        else :
            mov = 'nan'

    return mov

def open_move_cal(open_type):
    count = Counter(open_type)

    return count






