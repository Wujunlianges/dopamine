import os
import tensorflow as tf
from dopamine.agents.npg.npg_agent import NPGAgent
from dopamine.discrete_domains import run_experiment

import gin


def create_pg_agent(sess, environment, summary_writer=None):
    return NPGAgent(sess, num_actions=environment.action_space.n)


BASE_PATH = './tmp/pg'
LOG_PATH = os.path.join(BASE_PATH, 'npg_agent')

pg_config = """
import dopamine.discrete_domains.minigrid_lib
import dopamine.discrete_domains.run_experiment
import dopamine.replay_memory.circular_replay_buffer
import gin.tf.external_configurables

minigrid_lib.create_minigrid_environment.game_name = 'MiniGrid-Empty-5x5-v0'

run_experiment.Runner.create_environment_fn = @minigrid_lib.create_minigrid_environment
run_experiment.Runner.num_iterations = 500
run_experiment.Runner.training_steps = 1000
run_experiment.Runner.evaluation_steps = 1000
run_experiment.Runner.max_steps_per_episode = 200

circular_replay_buffer.WrappedReplayBuffer.replay_capacity = 50000
circular_replay_buffer.WrappedReplayBuffer.batch_size = 128


NPGAgent.observation_shape = %minigrid_lib.MINIGRID_OBSERVATION_SHAPE
NPGAgent.observation_dtype = %minigrid_lib.MINIGRID_OBSERVATION_DTYPE


tf.train.AdamOptimizer.learning_rate = 0.00001
tf.train.AdamOptimizer.epsilon = 0.00000390625
"""

tf.logging.set_verbosity(tf.logging.INFO)

gin.parse_config(pg_config, skip_unknown=False)

pg_runner = run_experiment.Runner(LOG_PATH, create_pg_agent)

print('Will train pg agent, please be patient, may be a while...')
pg_runner.run_experiment()
print('Done training!')