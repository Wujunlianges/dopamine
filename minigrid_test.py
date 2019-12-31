from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.discrete_domains import run_experiment

import tensorflow as tf

gin_files = ['dopamine/agents/dqn/configs/dqn.gin']
base_dir = './tmp/dqn/'
# gin_files = ['./minigrid.gin']
# base_dir = './tmp/minigrid/'
gin_bindings = []

tf.logging.set_verbosity(tf.logging.INFO)
run_experiment.load_gin_configs(gin_files, gin_bindings)
runner = run_experiment.create_runner(base_dir)
runner.run_experiment()