import numpy as np


def evaluate_model(agent, env, state_rms, eval_num=10):
    total_rewards = 0
    total_costs = 0
    for i in range(eval_num):
        s = env.reset()
        done = False
        while not done:
            s = state_rms(s)
            dist = agent.choose_dist(s)
            action = dist.sample().cpu().numpy()[0]

            next_state, reward, done, info = env.step(action)
            # env.render()
            s = next_state
            total_rewards += reward
            total_costs += info['cost']
    eval_rew = total_rewards / float(eval_num)
    eval_cost = total_costs / float(eval_num)
    return eval_rew, eval_cost


def evaluate_model_multi_seeds(agent, env, state_rms, eval_num=10):
    seeds = [1, 5, 8, 10, 20]
    total_rewards = 0
    for seed in seeds:
        env.seed(seed)
        for i in range(eval_num):
            s = env.reset()
            done = False
            while not done:
                s = state_rms(s)
                dist = agent.choose_dist(s)
                action = dist.sample().cpu().numpy()[0]

                next_state, reward, done, _ = env.step(action)
                # env.render()
                s = next_state
                total_rewards += reward
    eval_rew = total_rewards / (float(eval_num) * len(seeds))
    return eval_rew
