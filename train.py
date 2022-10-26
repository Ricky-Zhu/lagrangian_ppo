import torch
import numpy as np
import time
from ppo_utilities.running_mean_std import RunningMeanStd
from ppo_utilities.evaluation import evaluate_model
import wandb
from datetime import datetime
from tqdm import tqdm
from torch.nn.functional import softplus


class Train:
    def __init__(self, env, test_env, agent, args):
        self.env = env

        self.test_env = test_env
        self.agent = agent

        self.start_time = 0

        self.running_reward = 0
        self.env_steps = 0
        self.current_time = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')

        if args.start_train:
            wandb.login()
            wandb.init(
                project="{}".format(args.project_name),
                config=vars(args),
                name="{}_{}".format(args.env_name, self.current_time))
        self.args = wandb.config if args.start_train else args

        self.env_name = self.args.env_name
        self.epsilon = self.args.clip
        self.horizon = self.args.horizon
        self.epochs = self.args.epochs
        self.mini_batch_size = self.args.mini_batch_size
        self.n_iterations = self.args.n_iterations
        self.normalize_cost_advs = self.args.normalize_cost_advs
        self.cost_budget = self.args.cost_budget
        self.state_rms = RunningMeanStd(shape=(self.agent.n_states,), normalize_obs=self.args.normalize_obs)

    @staticmethod
    def choose_mini_batch(mini_batch_size, states, actions, returns, advs, values, cost_returns, cost_advs, cost_values,
                          log_probs):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices], returns[indices], advs[indices], values[indices], \
                  cost_returns[indices], cost_advs[indices], cost_values[indices], log_probs[indices]

    def train(self, states, actions, advs, values, cost_advs, cost_values, log_probs, avg_ep_cost):

        values = np.vstack(values[:-1])
        cost_values = np.vstack(cost_values[:-1])
        log_probs = np.vstack(log_probs)
        returns = advs + values
        cost_returns = cost_advs + cost_values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        if self.normalize_cost_advs:  # sometimes normalizing cost advantages can have a degraded performance
            cost_advs = (cost_advs - cost_advs.mean()) / (cost_advs.std() + 1e-8)

        actions = np.vstack(actions)
        cur_penalty = softplus(
            self.agent.penalty_param).item()  # fix the penalty value during the training in this iteration

        for epoch in range(self.epochs):
            for state, action, return_, adv, old_value, cost_return_, cost_adv, old_cost_value, old_log_prob in self.choose_mini_batch(
                    self.mini_batch_size,
                    states, actions, returns,
                    advs, values,
                    cost_returns, cost_advs,
                    cost_values, log_probs):
                state = torch.Tensor(state).to(self.agent.device)
                action = torch.Tensor(action).to(self.agent.device)
                return_ = torch.Tensor(return_).to(self.agent.device)
                adv = torch.Tensor(adv).to(self.agent.device)
                old_value = torch.Tensor(old_value).to(self.agent.device)
                cost_return_ = torch.Tensor(cost_return_).to(self.agent.device)
                cost_adv = torch.Tensor(cost_adv).to(self.agent.device)
                old_cost_value = torch.Tensor(old_cost_value).to(self.agent.device)
                old_log_prob = torch.Tensor(old_log_prob).to(self.agent.device)

                value = self.agent.critic(state)
                cost_value = self.agent.cost_critic(state)

                # clipped_value = old_value + torch.clamp(value - old_value, -self.epsilon, self.epsilon)
                # clipped_v_loss = (clipped_value - return_).pow(2)
                # unclipped_v_loss = (value - return_).pow(2)
                # critic_loss = 0.5 * torch.max(clipped_v_loss, unclipped_v_loss).mean()
                critic_loss = self.agent.critic_loss(value, return_)
                cost_critic_loss = self.agent.cost_critic_loss(cost_value, cost_return_)

                new_log_prob = self.calculate_log_probs(self.agent.current_policy, state, action)

                ratio = (new_log_prob - old_log_prob).exp()
                actor_loss = self.compute_actor_loss(ratio, adv, cost_adv, cur_penalty)
                penalty_loss = self.compute_penalty_loss(avg_ep_cost)

                self.agent.optimize(actor_loss, critic_loss, cost_critic_loss, penalty_loss)

        return actor_loss, critic_loss, cost_critic_loss, penalty_loss

    def step(self):
        state = self.env.reset()
        for iteration in tqdm(range(1, 1 + self.n_iterations)):
            states = []
            actions = []
            rewards = []
            costs = []
            values = []
            cost_values = []
            log_probs = []
            dones = []
            avg_ep_cost = 0.
            ep_count = 0

            self.start_time = time.time()
            for t in range(self.horizon):
                state = self.state_rms(state)
                dist = self.agent.choose_dist(state)
                action = dist.sample().cpu().numpy()[0]
                # action = np.clip(action, self.agent.action_bounds[0], self.agent.action_bounds[1])
                log_prob = dist.log_prob(torch.Tensor(action))
                value, cost_value = self.agent.get_value(state)
                next_state, reward, done, info = self.env.step(action)
                avg_ep_cost += info['cost']
                self.env_steps += 1

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                costs.append(info['cost'])
                values.append(value)
                cost_values.append(cost_value)
                log_probs.append(log_prob)
                dones.append(done)

                if done:
                    state = self.env.reset()
                    ep_count += 1
                else:
                    state = next_state
            # self.state_rms.update(next_state)
            next_state = self.state_rms(next_state)
            next_value, next_cost_value = self.agent.get_value(next_state)
            next_value *= (1 - done)
            next_cost_value *= (1 - done)
            values.append(next_value)
            cost_values.append(next_cost_value)
            avg_ep_cost /= float(ep_count)

            advs = self.get_gae(rewards, values, dones)
            cost_advs = self.get_gae(costs, cost_values, dones)
            states = np.vstack(states)
            actor_loss, critic_loss, cost_critic_loss, penalty_loss = self.train(states, actions, advs, values,
                                                                                 cost_advs, cost_values, log_probs,
                                                                                 avg_ep_cost)
            # self.agent.set_weights()
            self.agent.schedule_lr()
            eval_rewards, eval_costs = evaluate_model(self.agent, self.test_env, self.state_rms)

            self.state_rms.update(states)
            self.print_logs(iteration, actor_loss, critic_loss, eval_rewards, eval_costs)
            # wandb log the infos
            if self.args.start_train:
                wandb.log({'evaluation rewards': eval_rewards,
                           'evaluation running rewards': self.running_reward,
                           'evaluation costs': eval_costs,
                           'evaluation running costs': self.running_cost,
                           'loss/actor_loss': actor_loss.item(),
                           'loss/critic_loss': critic_loss.item()}, step=self.env_steps)
            # if self.running_reward > self.args.stopping_criterion:
            #     if self.args.start_train:
            #         self.agent.save_weights(iteration, self.state_rms)
            #     print('achieve rew:{}, criterion:{}, at iteration:{}'.format(eval_rewards, self.args.stopping_criterion,
            #                                                                  iteration))
            #     break

    @staticmethod
    def get_gae(rewards, values, dones, gamma=0.99, lam=0.95):

        advs = []
        gae = 0

        dones.append(0)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * (values[step + 1]) * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advs.append(gae)

        advs.reverse()
        return np.vstack(advs)

    @staticmethod
    def calculate_log_probs(model, states, actions):
        policy_distribution = model(states)
        return policy_distribution.log_prob(actions)

    def compute_penalty_loss(self, avg_ep_cost):
        penalty_loss = -self.agent.penalty_param * (avg_ep_cost - self.cost_budget)
        return penalty_loss

    def compute_actor_loss(self, ratio, adv, cost_adv, cur_penalty):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()

        # add the surrogate with cost advantages
        loss += cur_penalty * (ratio * cost_adv).mean()
        return loss

    def print_logs(self, iteration, actor_loss, critic_loss, eval_rewards, eval_costs):
        if iteration == 1:
            self.running_reward = eval_rewards
            self.running_cost = eval_costs
        else:
            self.running_reward = self.running_reward * 0.98 + eval_rewards * 0.02
            self.running_cost = self.running_cost * 0.98 + eval_costs * 0.02

        if iteration % 25 == 0:
            print(f"Iter:{iteration}| "
                  f"Ep_Reward:{eval_rewards:.3f}| "
                  f"Ep_Cost:{eval_costs:.3f}| "
                  f"Running_reward:{self.running_reward:.3f}| "
                  f"Running_cost:{self.running_cost:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Iter_duration:{time.time() - self.start_time:.3f}| "
                  f"lr:{self.agent.actor_scheduler.get_last_lr()}")
            self.agent.save_weights(iteration, self.state_rms)
