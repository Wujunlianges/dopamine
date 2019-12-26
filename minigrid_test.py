from gym_minigrid.wrappers import *
# env = gym.make('CartPole-v1')
env = gym.make('MiniGrid-Empty-8x8-v0')
env = RGBImgPartialObsWrapper(env)  # Get pixel observations
env = ImgObsWrapper(env)  # Get rid of the 'mission' field
obs = env.reset()  # This now produces an RGB tensor only
print('done')