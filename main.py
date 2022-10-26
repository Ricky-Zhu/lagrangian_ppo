import gym, safety_gym
import os
from safe_ppo_agent import Agent
from train import Train
import randomizer
from randomizer.wrappers import RandomizedEnvWrapper
from arguments import get_args
import os
from os.path import dirname
import randomizer.safe_env
from ppo_utilities.seeds import set_seeds
from ppo_utilities.evaluation import evaluate_model

TRAIN_FLAG = True

if __name__ == "__main__":
    args = get_args()

    ENV_NAME = args.env_name
    test_env = gym.make(ENV_NAME)

    # define the params for constructing the agent
    n_states = test_env.observation_space.shape[0]
    action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
    n_actions = test_env.action_space.shape[0]
    n_iterations = args.n_iterations
    lr = args.lr
    device = args.device

    print(f"number of states:{n_states}\n"
          f"action bounds:{action_bounds}\n"
          f"number of actions:{n_actions}")

    set_seeds(args)

    env = gym.make(ENV_NAME)
    env.seed(args.seed)
    test_env.seed(args.seed + 1)

    agent = Agent(n_states=n_states,
                  n_iter=n_iterations,
                  env_name=ENV_NAME,
                  action_bounds=action_bounds,
                  n_actions=n_actions,
                  lr=lr,
                  device=device)
    if TRAIN_FLAG:
        trainer = Train(env=env,
                        test_env=test_env,
                        agent=agent,
                        args=args)
        trainer.step()
    else:
        try:
            agent.load_weights()
        except:
            pass
        eval_rew, eval_cost = evaluate_model(agent, env, render=True)
        print(eval_rew, eval_cost)
    # player = Play(env, agent, ENV_NAME)
    # player.evaluate()
