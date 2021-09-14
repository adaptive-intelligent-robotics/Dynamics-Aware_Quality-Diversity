import gym
import panda_gym

#env = gym.make('PandaReach-v0', render=True)
#env = gym.make('PandaPush-v0', render=True)
#env = gym.make('PandaPickAndPlace-v0', render=True)
env = gym.make('BryanEnv-v0', render=True)


obs = env.reset()
done = False
for i in range(500):
    action = env.action_space.sample() # random action
    obs, reward, done, info = env.step(action)

    #print(action)
    print(obs['observation']) 
    #print(obs['achieved_goal'].shape)
    #print(obs['desired_goal'].shape)
    
    '''
    Action space: 
    print(action) - action is just a numpy array
    action space for all 3 envs is alwasy 4 
    x,y,z position of ee - 3 dim
    gripper (open or close) - 1 dim
    action is always -1 to 1 boundeed and then scaled to 0.05 for xyz
    gripper orientation is always zero

    Observation
    print(obs)
    observation is a dictionary

    # Reach env
    obs['observation'] - 10 dim 
    obs['achieved_goal'] - 3 dim 
    obs['desired_goal'] - 3 dim

    # Push env
    obs['observation'] - 25 dim 
    obs['achieved_goal'] - 3 dim 
    obs['desired_goal'] - 3 dim

    #Pick and Place env
    obs['observation'] - 25 dim 
    obs['achieved_goal'] - 3 dim 
    obs['desired_goal'] - 3 dim
    '''
env.close()
