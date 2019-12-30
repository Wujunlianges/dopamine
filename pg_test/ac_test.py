import os
import tensorflow as tf
from dopamine.agents.pg.ac import ACAgent
from dopamine.discrete_domains import run_experiment

import gin

def create_ac_agent(sess, environment, summary_writer=None):
    return ACAgent(sess, num_actions=environment.action_space.n)


BASE_PATH = './tmp/ac'
GAME = 'CartPole'
LOG_PATH = os.path.join(BASE_PATH, 'ac_agent', GAME)

ac_config = """
import dopamine.discrete_domains.gym_lib
import dopamine.discrete_domains.run_experiment
import dopamine.agents.dqn.dqn_agent
import dopamine.replay_memory.circular_replay_buffer
import gin.tf.external_configurables

create_gym_environment.environment_name = 'CartPole'
create_gym_environment.version = 'v0'
create_agent.agent_name = 'ac'

run_experiment.Runner.create_environment_fn = @gym_lib.create_gym_environment
run_experiment.Runner.num_iterations = 500
run_experiment.Runner.training_steps = 1000
run_experiment.Runner.evaluation_steps = 1000
run_experiment.Runner.max_steps_per_episode = 200

circular_replay_buffer.WrappedReplayBuffer.replay_capacity = 50000
circular_replay_buffer.WrappedReplayBuffer.batch_size = 128

ACAgent.actor_network = @CartpoleActorNetwork
ACAgent.critic_network = @CartpoleCriticNetwork
ACAgent.observation_shape = %gym_lib.CARTPOLE_OBSERVATION_SHAPE
ACAgent.observation_dtype = %gym_lib.CARTPOLE_OBSERVATION_DTYPE
ACAgent.stack_size = %gym_lib.CARTPOLE_STACK_SIZE


tf.train.AdamOptimizer.learning_rate = 0.00001
tf.train.AdamOptimizer.epsilon = 0.00000390625
"""

tf.logging.set_verbosity(tf.logging.INFO)

gin.parse_config(ac_config, skip_unknown=False)

ac_runner = run_experiment.Runner(LOG_PATH, create_ac_agent)

print('Will train ac agent, please be patient, may be a while...')
ac_runner.run_experiment()
print('Done training!')