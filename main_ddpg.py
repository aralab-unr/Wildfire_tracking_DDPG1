import gym
import numpy as np
from ddpg_torch import Agent
from utils import plot_learning_curve
from fire_model_test import FireEnvironment

if __name__ == '__main__':

    fire_input = "fbndry4.txt"
    dim_x = 15
    dim_y = 15
    dim_z = 5
    agents_theta_degrees = 30
    n_drones = 3
    X_MAX = 14
    X_MIN = 0
    Y_MAX = 14
    Y_MIN = 0
    Z_MAX = 4
    Z_MIN = 1

    outputs_dir = "results"
    n_episodes = 1005
    episode_steps = np.zeros(n_episodes).astype(int)
    fov_angle = np.array([30, 30])
    total_reward = 0
    total_rewards = np.zeros(n_episodes).astype(float)

    env = FireEnvironment("fbndry4.txt", dim_x, dim_y, [dim_x, dim_y, dim_z], agents_theta_degrees, n_drones, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX)

    agent = Agent(alpha = 0.0001, beta = 0.001, input_dims = (n_drones * 3), tau = 0.001,
            batch_size = 64, fc1_dims = 400, fc2_dims = 300, n_actions = (n_drones * 6))

    filename = 'Wildfire_alpha_' + str(agent.alpha) + '_beta_' + \
                str(agent.beta) + '_' + str(n_episodes) + '_games'
    figure_file = 'plots/' + filename + '.png'

    best_score = env.reward_range[0]
    score_history = []
    reward_history = []


    for i in range(n_episodes):
        simulation_index = 0
        env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            env.map = env.simStep(simulation_index)
            simulation_index+=1
            observation = env.state()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            #print("Step ",env.steps, " - ",info)

            agent.remember(observation, action, reward, observation_, done)
            score += reward
            observation = observation_

            agent.learn()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    x = [i+1 for i in range(n_episodes)]
    plot_learning_curve(x, score_history, figure_file)
