import gym
import torch
import numpy as np

from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from replay_buffer import ReplayBuffer, Transition

DEVICE = torch.device('cuda:0')
DTYPE = torch.float

args = {'env_name': 'MountainCarContinuous-v0',
        'seed': 42,
        'gamma': 0.9,
        'tau': 0.001,
        'hidden_size': 128,
        'replay_size': 1000000,
        'num_episodes': 1000,
        'batch_size': 128,
        'replay_num_updates': 5,
        'ou_noise': True,
        'noise_scale': 0.3,
        'final_noise_scale': 0.3,
        'exploration_end': 100}

env = NormalizedActions(gym.make(args['env_name']))

env.seed(args['seed'])
torch.manual_seed(args['seed'])
np.random.seed(args['seed'])

agent = NAF(args['gamma'], args['tau'], args['hidden_size'],
            env.observation_space.shape[0], env.action_space)
agent.load_model(f'models/naf_{args["env_name"]}_')

replay_buffer = ReplayBuffer(args['replay_size'])

ounoise = OUNoise(env.action_space.shape[0]) if args['ou_noise'] else None


def run():
    num_steps = 0
    rewards = []

    for episode in range(args['num_episodes']):
        state = torch.Tensor([env.reset()]).to(device=DEVICE)
        set_noise(episode)
        episode_reward = 0
        episode_steps = 0
        is_done = False

        while not is_done:
            act = agent.select_action(state, ounoise)
            suc_state, reward, is_done, _ = env.step(act.numpy()[0])
            num_steps += 1
            episode_steps += 1
            if episode_steps % 100 == 0:
                print(episode_steps)
            episode_reward += reward

            # prepare for insert to replay buffer
            # action = torch.Tensor(act).to(device=DEVICE)
            # done = torch.tensor([not is_done]).to(device=DEVICE)  # put 0 if is done
            # successor_state = torch.Tensor([suc_state]).to(device=DEVICE)
            # reward = torch.Tensor([reward]).to(device=DEVICE)

            action = np.array(act)
            done = np.array([not is_done])
            successor_state = np.array([suc_state])
            reward = np.array([reward])
            replay_buffer.push(state, action, done, successor_state, reward)

            state = successor_state

            if len(replay_buffer) > args['batch_size']:
                sample_buffer()

        rewards.append(episode_reward)
        print(rewards)

        if episode % 2 == 0:
            episode_reward = run_simulation()
            rewards.append(episode_reward)
            print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(episode, num_steps,
                                                                                           rewards[-1],
                                                                                           np.mean(rewards[-10:])))
        if episode % 5 == 0:
            print('save model')
            agent.save_model(args['env_name'])

    env.close()


def run_simulation():
    state = torch.Tensor([env.reset()]).to(device=DEVICE)
    episode_reward = 0
    while True:
        action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action.numpy()[0])
        episode_reward += reward

        next_state = torch.Tensor([next_state]).to(device=DEVICE)

        state = next_state
        if done:
            break

    return episode_reward


def set_noise(episode):
    if ounoise:
        ounoise.scale = (args['noise_scale'] - args['final_noise_scale']) * max(0, args['exploration_end'] -
                                                                                episode) / args['exploration_end'] + \
                        args['final_noise_scale']
        ounoise.reset()


def sample_buffer():
    for i in range(args['replay_num_updates']):
        transitions = replay_buffer.sample(args['batch_size'])
        batch = Transition(*zip(*transitions))
        val_loss = agent.update_parameters(batch)
        # print(val_loss)


if __name__ == '__main__':
    run()
