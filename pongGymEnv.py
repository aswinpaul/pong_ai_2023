"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
# from readState import Ball, Paddle, Rectangle, Point, AREA_WIDTH, AREA_HEIGHT, Ball_prediction
from readState import Pong_config, Pong_state, Main_Config, init_pong_state, pong_state_step,Env_config,Spike_config,Stim_config,Sensory_stim_config,Random_stim_config,Proprio_stim_config,Stim_channel_config,Pred_stim_config, Baseline_config, Stim_seq_ctrl, Stim_freq_ctrl
import dill
import subprocess

class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        # self.ball0 = Ball(position=Point(x=0,y=0), velocity=Point(0,0))
        # self.paddle0 = Paddle(position=Rectangle(top = Point(x=0,y=0), bottom=Point(x=0,y=0)),velocity=Point(x=0, y=0))
        # self.paddle1 = Paddle(position=Rectangle(top = Point(x=0,y=0), bottom=Point(x=0,y=0)),velocity=Point(x=0, y=0))
        # self.game_area = Rectangle(top=Point(x=0, y=0),bottom=Point(x=AREA_WIDTH, y=AREA_HEIGHT))
        # self.hit_count = 0
        # self.miss_count = 0
        # self.phase = 0
        # self.rally_length = 0
        # self.left_prediction = Ball_prediction(position=Point(x=0,y=0),paddle_centre=0.0, frame_no=0)
        # self.right_prediction = Ball_prediction(position=Point(x=0,y=0),paddle_centre=0.0, frame_no=0)
        self.justonce = 0
        self.frame_no = 0
        self.elapse = 0
        self.f = open('test.out', 'wb')
        self.game_config = Pong_config()
        self.game_state = Pong_state()
        self.main_config = Main_Config(env=Env_config(open_loop=False, pong=self.game_config),
                            spike=Spike_config(dummy=0),
                            stim=Stim_config(sensory=Sensory_stim_config(amp=0,phase=0,min_frequency=0,max_frequency=0,enabled=False),
                            random=Random_stim_config(amp=0,phase=0, interval=0,count=0, cooldown=0,pulse_count=0, pulse_interval=0, enabled=False),
                            pred=Pred_stim_config(amp=0,phase=0,pulse_count=0,pulse_interval=0,enabled=False),proprio=Proprio_stim_config(enabled=False),channel=Stim_channel_config(id=[], num=0)),
                            baseline=Baseline_config(dummy=0))
        self.stim_seq_ctrl = Stim_seq_ctrl(n_seq=0, seq=[],seq_start_tick=[], seq_next_idx=[], last_tick=0)
        self.stim_freq_ctrl = Stim_freq_ctrl(amp=0,phase=0,last_tick=0,spec=[],primary_unit=0)
        init_pong_state(self.game_config, self.game_state, self.frame_no)
        dill.dump(self.game_state, self.f)
        self.state1 = [self.game_state.ball0.position.x,
                      self.game_state.ball0.position.y,
                      self.game_state.ball0.velocity.y,
                      self.game_state.paddle0.position.top.y,
                      self.game_state.paddle0.position.bottom.y]

        # print(self.game_state)


        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        # self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print(">>here", action)
        # paddle_movement = random.choice([0,2,-2])
        if action == 0:
            paddle_movement = 10.0
        elif action == 1:
            paddle_movement = -10.0
        elif action ==2:
            paddle_movement = 0
        else:
            raise Exception("WTF")
        # paddle_movement = 0
        pong_state_step(self.game_state, self.main_config, self.frame_no, self.elapse, paddle_movement,self.stim_seq_ctrl, self.stim_freq_ctrl)
        self.frame_no+=1
        self.elapse+=2
        self.state1 = [self.game_state.ball0.position.x,
                      self.game_state.ball0.position.y,
                      self.game_state.ball0.velocity.y,
                      self.game_state.paddle0.position.top.y,
                      self.game_state.paddle0.position.bottom.y]
        done = bool(
            self.game_state.miss_count > 0
        )
        # print(self.state1, self.game_state.miss_count, done, self.steps_beyond_done)
        self.game_state.frame+=1
        dill.dump(self.game_state, self.f)
        # if done:
        #     print("Ball missed!!")
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state1, dtype=np.float32), self.game_state.hit_count, done, {}

    def reset(self, seed):
        init_pong_state(self.game_config, self.game_state, self.frame_no)
        self.game_state.miss_count = 0
        self.steps_beyond_done = None
        self.game_state.seed = seed
        return np.array(self.state1, dtype=np.float32)
    
    def render(self, mode='rgb_array'):
        if self.state1 is None:
            return None
        return self.state1

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
            
    def board_constants(self):
        return self.game_state.game_area.top.x, self.game_state.game_area.bottom.x, self.game_state.game_area.bottom.y, self.game_state.game_area.top.y, self.game_state.paddle.position.bottom.x, self.game_state.paddle.position.top.x