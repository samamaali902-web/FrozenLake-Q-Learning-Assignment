import gymnasium as gym
import numpy as np
from agent import QLearningAgent
from utils import plot_progress, save_gif

EPISODES = 15000
SEED = 42

def train():
    env = gym.make("FrozenLake-v1", is_slippery=False)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, seed=SEED)
    rewards=[]
    for ep in range(EPISODES):
        state,_ = env.reset(seed=SEED+ep)
        total=0
        for _ in range(100):
            action=agent.act(state)
            next_state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
            agent.learn(state,action,reward,next_state,done)
            state=next_state
            total+=reward
            if done: break
        agent.decay()
        rewards.append(total)
        if (ep+1)%3000==0:
            print(f"Episode {ep+1}: last 200 success rate={np.mean(rewards[-200:]):.2%}")
    np.save("q_table.npy",agent.q)
    plot_progress(rewards)
    env.close()
    return agent

def evaluate(agent, episodes=100):
    env=gym.make("FrozenLake-v1",is_slippery=False)
    wins=0
    for ep in range(episodes):
        state,_=env.reset(seed=100000+ep)
        for _ in range(100):
            action=agent.act(state,training=False)
            state,reward,terminated,truncated,_=env.step(action)
            if terminated or truncated:
                wins+=int(reward>0)
                break
    env.close()
    print(f"Evaluation success: {wins}/{episodes} ({wins/episodes:.1%})")

def demo(agent):
    env=gym.make("FrozenLake-v1",is_slippery=False)
    state,_=env.reset(seed=SEED)
    states=[state]
    for _ in range(100):
        state,reward,terminated,truncated,_=env.step(agent.act(state,training=False))
        states.append(state)
        if terminated or truncated: break
    env.close()
    save_gif(states)
    print(f"GIF saved, goal reached: {reward>0}")

if __name__=="__main__":
    agent=train()
    evaluate(agent)
    demo(agent)
