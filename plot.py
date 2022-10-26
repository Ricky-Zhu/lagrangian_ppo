from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt.util import load_logs
from matplotlib import gridspec


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size': 30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]),
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% confidence interval')

    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size': 20})
    axis.set_xlabel('x', fontdict={'size': 20})

    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size': 20})
    acq.set_xlabel('x', fontdict={'size': 20})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def black_box_function(cart, pole):
    return cart ** 2 + pole ** 2


# define the BO
pbounds = {'cart': (0.01, 0.30), 'pole': (0.02, 0.07)}
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=7
)
log_path = '/home/ruiqi/bayrn_logs_InvertedDoublePendulumRandomizedEnv-v0.json'

load_logs(optimizer, logs=[log_path])
print('successful load previous log!')

cart = np.array([[res["params"]["cart"]] for res in optimizer.res]).round(5)
pole = np.array([[res["params"]["pole"]] for res in optimizer.res]).round(5)
target = np.array([res["target"] for res in optimizer.res])
input = np.concatenate([cart, pole], axis=1)
optimizer._gp.fit(input, target)




resolution = 1000

cart1 = np.linspace(0.01, 0.30, resolution).round(5)
pole1 = np.linspace(0.02, 0.07, resolution).round(5)
cart2, pole2 = np.meshgrid(cart1, pole1)
input1 = np.concatenate([cart2[:, :, None], pole2[:, :, None]], axis=2).reshape(-1,2)

mu, sigma = optimizer._gp.predict(input1, return_std=True)
mu = mu.reshape(resolution,resolution)
sigma = sigma.reshape(resolution,resolution)
ax = plt.axes(projection='3d')
ax.view_init(90, 0)
ax.plot_surface(cart2, pole2, mu,
                cmap='viridis', edgecolor='none')
ax.scatter(cart,pole,target,marker='*',c='red',alpha=1)
default = np.array([0.10,0.045]).reshape(-1,2)

print(optimizer._gp.predict(default, return_std=True)[0])
plt.show()
