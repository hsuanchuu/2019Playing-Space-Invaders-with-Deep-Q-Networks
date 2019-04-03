import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="DQN: Atari 2600")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--do_render', action='store_true', help='whether display video in testing mode')

    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env_name = args.env_name or 'SpaceInvadersNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        env = Environment('SpaceInvadersNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)
