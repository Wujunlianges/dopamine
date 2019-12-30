from multiprocessing import Process, Pipe
import gym
import gin


def AtariWorker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "step_with_loss_life_terminal":
            old_lives = env.ale.lives()
            obs, reward, done, info = env.step(data)
            new_lives = env.ale.lives()
            done = done or new_lives < self.lives
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "lives":
            lives = env.ale.lives()
            conn.send(lives)
        else:
            raise NotImplementedError


@gin.configurable
class ParallelAtariPreprocessing:
    def __init__(self,
                 envs,
                 frame_skip=4,
                 terminal_on_life_loss=False,
                 screen_size=84):
        if frame_skip <= 0:
            raise ValueError(
                'Frame skip should be strictly positive, got {}'.format(
                    frame_skip))
        if screen_size <= 0:
            raise ValueError(
                'Target screen size should be strictly positive, got {}'.
                format(screen_size))
        if len(envs) <= 0:
            raise ValueError('No environment given.')

        self.envs = envs
        self.procs = len(envs)
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_dims = self.envs[0].observation_space
        self.screen_buffer = [[
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ] for i in range(self.procs)]

        self.game_over = False
        self.lives = 0

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=AtariWorker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    @property
    def observation_space(self):
        return Box(low=0,
                   high=255,
                   shape=(self.screen_size, self.screen_size, 1),
                   dtype=np.uint8)

    @property
    def action_space(self):
        return self.envs[0].action_space

    @property
    def reward_range(self):
        return self.envs[0].reward_range

    @property
    def metadata(self):
        return self.envs[0].metadata

    def close(self):
        return self.envs[0].close()

    def reset(self):
        # return a list
        for local in self.locals:
            local.send(('reset', None))
        observations = [self.envs[0].reset()[0]
                        ] + [local.recv() for local[0] in self.locals]
        self.lives = self.envs[0].ale.lives()
        self.save_grayscale_obs(observations, 0)
        for i in range(self.procs):
            self.screen_buffer[i][1].fill(0)
        return self._pool_and_resize()

    def render(self, mode):
        raise NotImplementedError

    def step(self, actions):
        # return 4 tuple
        accumulated_rewards = [0. for i in range(self.procs)]
        for time_step in range(self.frame_skip):
            observation, reward, game_over, info = self.envs[0].step(
                actions[0])
            accumulated_rewards[0] += reward
            observations.append(observation)
            game_overs.append(game_over)
            infos.append(info)

            if self.terminal_on_life_loss:
                for local, action in zip(self.locals, actions[1:]):
                    local.send(('step_with_loss_life_terminal', action))
                new_lives = self.envs[0].ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                for local, action in zip(self.locals, actions[1:]):
                    local.send(('step', action))
                is_terminal = game_over

            results = [(observation, reward, is_terminal, info)
                       ] + [local.recv() for local in self.locals]
            for i in range(self.procs):
                accumulated_rewards[i] += results[1][i]

            if not all(is_terminal == False for is_terminal in results[2]):
                break
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self.save_grayscale_obs(results[0], t)

        self.game_over = game_over
        return observation, accumulated_rewards, results[2], results[3]

    def save_grayscale_obs(self, observations, t):
        for i in range(self.procs):
            self.screen_buffer[i][t] = np.dot(observations[i][..., :3],
                                              [0.299, 0.587, 0.114])

    def _pool_and_resize(self):
        int_images = []
        for i in range(self.procs):
            if self.frame_skip > 1:
                np.maximum(self.screen_buffer[i][0],
                           self.screen_buffer[i][1],
                           out=self.screen_buffer[i][0])

            transformed_image = cv2.resize(
                self.screen_buffer[i][0], (self.screen_size, self.screen_size),
                interpolation=cv2.INTER_AREA)
            int_image = np.asarray(transformed_image, dtype=np.uint8)
            int_images.append(np.expand_dims(int_image, axis=2))
        return int_images


@gin.configurable
def create_parallel_atari_environment(game_name, procs=16,
                                      sticky_actions=True):
    assert game_name is not None
    game_version = 'v0' if sticky_actions else 'v4'
    full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
    envs = []
    for i in range(procs):
        envs.append(gym.make(full_game_name))
    envs = ParallelAtariPreprocessing(envs)
    return envs
