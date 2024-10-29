import gym
import numpy as np

from dqn_agent import DQNAgent


def main():
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Load the pre-trained model if needed
    try:
        agent.load("dqn_model.pth")  # Uncomment to load a saved model
    except Exception as e:
        print(f"Could not load model: {e}")

    episodes = 1000
    batch_size = 32
    
    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        print(f"Starting Episode: {e + 1}")  # Debug print
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Episode: {e + 1}/{episodes}, Score: {time + 1}, Epsilon: {agent.epsilon:.2f}")  # Debug print
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    
    # Save the trained model
    agent.save("dqn_model.pth")
    print("Model saved!")  # Debug print

    env.close()

if __name__ == "__main__":
    main()