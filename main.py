import gym
import torch
import numpy as np

from naf import NAF
from normalized_actions import NormalizedActions
from replay_buffer import ReplayBuffer, Transition

args = {'env_name': 'MountainCarContinuous-v0',
        'seed': 42,
        'gamma': 0.9,
        'tau': 0.001,
        'hidden_size': 128,
        'replay_size': 1000000,
        'num_episodes': 1000,
        'batch_size': 128,
        'replay_num_updates': 5}

env = NormalizedActions(gym.make(args['env_name']))

env.seed(args['seed'])
torch.manual_seed(args['seed'])
np.random.seed(args['seed'])

agent = NAF(args['gamma'], args['tau'], args['hidden_size'],
            env.observation_space.shape[0], env.action_space)

replay_buffer = ReplayBuffer(args['replay_size'])

# TODO: mkae here some noise (add noise)

num_steps = 0
rewards = []

for episode in range(args['num_episodes']):
    state = torch.Tensor([env.reset()])

    # TODO: make some noise here also

    episode_reward = 0

    while True:
        act = agent.select_action(state, None) # TODO: replace none with noise
        suc_state, reward, is_done, _ = env.step(act.numpy()[0])
        num_steps += 1
        episode_reward += reward

        # prepare for insert to replay buffer
        action = torch.Tensor(act)
        done = torch.tensor([not is_done])  # put 0 if is done
        successor_state = torch.Tensor([suc_state])
        reward = torch.Tensor([reward])

        replay_buffer.push(state, action, done, successor_state, reward)

        state = successor_state

        if len(replay_buffer) > args['batch_size']:
            for i in range(args['replay_num_updates']):
                transitions = replay_buffer.sample(args['batch_size'])
                batch = Transition(*zip(*transitions))

                val_loss = agent.update_parameters(batch)

        if is_done:
            break

        # TODO Update param_noise based on distance metric

        rewards.append(episode_reward)

        if episode % 10 == 0:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])
                episode_reward += reward

                next_state = torch.Tensor([next_state])

                state = next_state
                if done:
                    break

            rewards.append(episode_reward)
            print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(episode, num_steps,
                                                                                           rewards[-1],
                                                                                           np.mean(rewards[-10:])))
env.close()