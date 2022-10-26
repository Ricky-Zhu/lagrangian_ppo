import argparse

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--project-name', type=str, default='ppo', help='define the project name')
    parse.add_argument('--device', type=str, default='cpu')
    parse.add_argument('--seed', type=int, default=123, help='the random seeds')
    parse.add_argument('--env-name', type=str, default='RandomizeSafeDoublePendulum-v0', help='the environment name')
    parse.add_argument('--n-iterations', type=int, default=501)
    parse.add_argument('--lr', type=float, default=3e-4, help='learning rate of the algorithm')
    parse.add_argument('--epochs', type=int, default=10, help='the epoch during training')
    parse.add_argument('--horizon', type=int, default=2048, help='the steps to collect samples')
    parse.add_argument('--clip', type=float, default=0.2, help='the ratio clip param')
    parse.add_argument('--mini-batch-size', type=int, default=64)
    parse.add_argument('--stopping-criterion', type=float, default=900.)
    parse.add_argument('--cost-budget', type=float, default=30.)

    parse.add_argument('--start-train', action='store_true')
    parse.add_argument('--normalize-obs', action='store_true')
    parse.add_argument('--randomize-domain', action='store_true')
    parse.add_argument('--normalize-cost-advs', action='store_true')

    args = parse.parse_args()

    return args
