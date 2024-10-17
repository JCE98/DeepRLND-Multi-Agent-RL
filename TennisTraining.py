import torch, argparse, warnings, time, os
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from ddpg_agent import Agent
from collections import deque

def argparser():
    parser = argparse.ArgumentParser(description="Parse arguments for DDPG agent training on Tennis application")
    parser.add_argument("--training_episodes", type=int, nargs='?', default=3000, help="number of episodes to train agent")
    parser.add_argument("--max_iterations", type=int, nargs='?', default=700, help="maximum number of iterations to run each episode")
    parser.add_argument("--buffer_size", type=int, nargs='?', default=int(1e6), help="replay buffer size")
    parser.add_argument("--batch_size", type=int, nargs='?', default=128, help="minibatch size")
    parser.add_argument("--gamma", type=float, nargs='?', default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, nargs='?', default=1e-3, help="for soft update of target parameters")
    parser.add_argument("--actor_lr", type=float, nargs='?', default=1e-4, help="actor learning rate")
    parser.add_argument("--critic_lr", type=float, nargs='?', default=3e-4, help="critic learning rate")
    parser.add_argument("--weight_decay", type=float, nargs='?', default=0.0001, help="L2 weight decay")
    args = parser.parse_args()
    return args

def ddpg(env, args):
    print("Performing environment setup and agent generation...")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]                                          # resetting environment to obtain state and action spaces
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {} to produce an action with length {}'.format(states.shape[0], state_size, action_size))
    scores_deque = deque(maxlen=100)
    scores = np.array([])
    max_score = -np.Inf
    agent = Agent(state_size, action_size, num_agents, args, random_seed=10)
    for i_episode in range(1, args.training_episodes+1):
        start = time.time()
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(args.max_iterations):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += rewards
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            if np.any(dones):
                break 
        max_score = np.max(score)
        scores_deque.append(max_score)
        scores = np.append(scores,max_score)
        print('Episode {}\tAverage Score: {:.2f}\tEpisode Time: {:.2f}'.format(i_episode, np.mean(scores_deque), (time.time()-start)), end="\r")
        if i_episode % 100 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\nAverage Score of Last 100 episodes: {:.2f}'.format(np.mean(scores_deque)))
        if len(scores_deque) == 100 and np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes.'.format(i_episode))
            torch.save(agent.actor_local.state_dict(), 'solution_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'solution_critic.pth')
            break   
        if (i_episode==args.training_episodes):
            print('\nMax training episodes reached without environment solution!')
    return scores

def test_agent(env, episodes=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")             # tensor casting device
    tensor_kwargs = {'device':device, 'dtype':torch.float32, 'requires_grad':True}      # keyword arguments for tensor instatntiation
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    scores_array = np.array([])                                                         # container to capture mean scores from each episode for plotting
    agent = Agent(state_size, action_size, num_agents, args, random_seed=10)            # instantiate agent object
    # Load agent solution
    agent.actor_local.load_state_dict(torch.load('solution_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('solution_critic.pth'))

    scores = []

    for i_episode in range(1, episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        agent.reset()
        score = np.zeros(num_agents)                          # initialize the score (for each agent)

        while True:
            actions = agent.act(states)                        # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to the environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            score += rewards                                   # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if any(dones):
                break

        mean_score = np.mean(score)
        scores_array = np.append(scores_array,mean_score)
        # Print out Score for each testing episode
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))

    # Compute the overall mean score from testing and print out resutls
    overall_mean = np.mean(scores)
    print('\nAverage Score {:.2f} Episodes: {:.2f}'.format(overall_mean, i_episode))
            
    return scores_array

if __name__=="__main__":
    warnings.simplefilter('ignore', UserWarning)                                    # prevents torch.nn.functional.tanh deprecation warning
    # command line arguments
    print('Parsing command line arguments...')
    args = argparser()
    # environment setup
    print('Setting up environment...')
    path = "C:/Users/josia/Documents/Education/Udacity_Nanodegrees/Udacity_Deep_RL_Nanodegree/Multi-Agent_Reinforcement_Learning/Project/DeepRLND-Multi-Agent-RL/Tennis_Windows_x86_64/Tennis.exe"
    env = UnityEnvironment(file_name=path)
    # train agent
    print('Entering training loop...')
    train_scores = ddpg(env, args)
    print('Training Complete!')
    # plot training results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(train_scores)+1), train_scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('avg_train_score.png')
    plt.show()
    # test trained agent
    if os.path.exists("./solution_actor.pth"):                                      # check whether solution has been found
        # test trained agent
        print('Testing trained agent...')
        test_scores = test_agent(env)
        print('Testing complete!')
        # plot testing results
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(test_scores)+1), test_scores)                     # plot average scores per episode
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig('avg_test_score.png')
        plt.show()
    print("Closing Unity environment...")
    env.close()