from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
from typing import List
import random
import math
import time
import mmap
import time
import copy
import pandas as pd

# from _env_stream import ffi, lib

# from _env_stream.lib import MEA_PONG_NORMAL_PLAY, MEA_PONG_RANDOM_SEQUENCE, MEA_ENV_PONG, MEA_ENV_JUMP, MEA_JUMP_RANDOM_SEQUENCE


class Pong_ball_event(Enum):
	BALL_NO_EVENT = 0
	BALL_BOUNCE_WALL = 1
	BALL_BOUNCE_PADDLE = 2
	BALL_BOUNCE_OPPONENT = 3
	BALL_MISSED = -1

class Phase(Enum):
    NORMAL_PLAY = 0
    RANDOM_SEQUENCE = 1
    PRED_SEQUENCE = 2

class Mea_decoder_type(Enum):
    MEA_DECODER_SUM = 0
    MEA_DECODER_SUM_EXP3 = 1
    MEA_DECODER_LINEAR = 2

class Mea_lin_dec_alg(Enum):
    MEA_LIN_DEC_NONE = 0
    MEA_LIN_DEC_CLONE = 1
    MEA_LIN_DEC_REWARD_SHAPING = 2

class Mea_lin_dec_init(Enum):
    MEA_LIN_DEC_ZERO = 0
    MEA_LIN_DEC_MOTOR_REGIONS = 1

class Mea_lin_dec_clone_behaviour(Enum):
    MEA_LIN_DEC_CLONE_SIMPLE = 0
    MEA_LIN_DEC_CLONE_RELAXED = 1

class Mea_lin_dec_clone_output(Enum):
    MEA_LIN_DEC_CLONE_IMPULSE = 0
    MEA_LIN_DEC_CLONE_VELOCITY = 1

class Mea_lin_dec_clone_loss(Enum):
    MEA_LIN_DEC_CLONE_STRICT = 0
    MEA_LIN_DEC_CLONE_HINGE = 1
    MEA_LIN_DEC_CLONE_HINGE_NO_STILL = 2

class Pong_side(Enum):
   PONG_SIDE_LEFT = 0 
   PONG_SIDE_RIGHT = 1

class Point:
    def __init__(self, x=0,y=0):
       self.x = x
       self.y = y
# @dataclass
# class Point:
#     x: float
#     y: float

class Rectangle:
    def __init__(self) -> None:
        self.top = Point(0,0)
        self.bottom = Point(0,0)
# @dataclass
# class Rectangle:
#     top:Point
#     bottom:Point

class Ball:
    def __init__(self) -> None:
        self.position = Point(0,0)
        self.velocity = Point(0,0)
# @dataclass
# class Ball:
#     position: Point
#     velocity: Point

class Paddle:
    def __init__(self) -> None:
        self.position = Rectangle()
        self.velocity = Point()
# @dataclass
# class Paddle:
#     position: Rectangle
#     velocity: Point

class Ball_prediction:
    def __init__(self) -> None:
        self.position = Point(0,0)
        self.paddle_centre: float = 0.0
        self.frame_no: np.int64 = 0.0
# @dataclass
# class Ball_prediction:
#     position: Point
#     paddle_centre: float
#     frame_no: np.int64

class Pong_state:
    def __init__(self) -> None:
        self.frame = 0
        self.ball0 = Ball()
        self.paddle0 = Paddle()
        self.paddle1 = Paddle()
        self.game_area = Rectangle()
        self.hit_count = 0
        self.miss_count = 0
        self.phase = 0
        self.rally_length:int = 0
        self.left_prediction = Ball_prediction()
        self.right_prediction = Ball_prediction()
        self.seed = 0
# @dataclass
# class Pong_state:
#     ball0: Ball
#     paddle0: Paddle
#     paddle1: Paddle
#     game_area:Rectangle
#     hit_count:int
#     miss_count:int
#     phase:Phase
#     rally_length:int
#     left_prediction:Ball_prediction
#     right_prediction:Ball_prediction

@dataclass
class Baseline_state:
	m1_baseline:float
	m2_baseline: float
	m1_last_gain: float
	m2_last_gain: float

# @dataclass
# class Env_state:
#     clock: np.int64
#     frame_no: np.int64
#     env_name: int
#     env: State
#     baseline_state:Baseline_state
#     start_frame: np.int64

class Pong_config:
    def __init__(self) -> None:
        self.decoder = 1
        self.paddle_size = 250
        self.friction = 0.1
        self.lin_dec_alg = 0
        self.lin_dec_init = 0
        self.lin_dec_clone_behaviour = 0
        self.lin_dec_clone_output = 0
        self.lin_dec_clone_loss = 0
        self.lin_dec_learning_rate = None
        self.lin_dec_clone_speed_cap = None
        self.exp3_eta = None

# @dataclass
# class Pong_config:
#     decoder: Mea_decoder_type
#     paddle_size: float
#     friction: float
#     lin_dec_alg: Mea_lin_dec_alg
#     lin_dec_init: Mea_lin_dec_init
#     lin_dec_clone_behaviour: Mea_lin_dec_clone_behaviour
#     lin_dec_clone_output: Mea_lin_dec_clone_output
#     lin_dec_clone_loss: Mea_lin_dec_clone_loss
#     lin_dec_learning_rate: float
#     lin_dec_clone_speed_cap: float
#     exp3_eta: float

# TODO
@dataclass
class Env_config:
    open_loop: bool
    pong: Pong_config

# TODO
@dataclass
class Spike_config:
    dummy:int

@dataclass
class Sensory_stim_config:
	amp:int
	phase:int
	min_frequency:float
	max_frequency:float
	enabled: bool

@dataclass
class Random_stim_config:
	amp: int
	phase: int
	interval: int
	count: int
	cooldown: int
	pulse_count: int
	pulse_interval: int
	enabled: bool

@dataclass
class Pred_stim_config:
	amp: int
	phase: int
	pulse_count: int
	pulse_interval: int
	enabled: bool

@dataclass
class Proprio_stim_config:
	enabled: bool

@dataclass
class Stim_channel_config:
	id: List[str]
	num: int

@dataclass
class Stim_config:
    sensory: Sensory_stim_config
    random: Random_stim_config
    pred: Pred_stim_config
    proprio: Proprio_stim_config
    channel: Stim_channel_config

# TODO
@dataclass
class Baseline_config:
    dummy:int

@dataclass
class Main_Config:
    env: Env_config
    spike: Spike_config
    stim: Stim_config
    baseline: Baseline_config

@dataclass
class Stim_pulses:
    count: int
    interval: int

@dataclass
class Stim_cfg:
    unit: int
    amp: int
    phase: int
    pulses: Stim_pulses

@dataclass
class Stim_seq:
    n_stim: int
    stim: List[Stim_cfg]
    ts: List[np.int64]

@dataclass
class Stim_seq_ctrl:
    n_seq:int
    seq: List[Stim_seq]
    seq_start_tick: List[np.int64]
    seq_next_idx: List[int]
    last_tick: np.int64

@dataclass
class Stim_freq_spec:
    value: int
    interval: int
    pulses: Stim_pulses

@dataclass
class Stim_freq_ctrl:
    amp: int
    phase: int
    last_tick: np.int64
    spec: List[Stim_freq_spec]
    primary_unit: int

STATE_STEP_INTERVAL = 20
MEA_NCHANNEL = 1024
MEA_CHANNEL_M1 = 1
MEA_CHANNEL_M2 = 2
RAND_SEQ_IDX = 0
AREA_HEIGHT = 600
AREA_WIDTH = 600
PADDLE_WIDTH = 10
PONG_SIDE_LEFT = 0
PONG_SIDE_RIGHT = 1
PRED_SEQ_IDX = 0
SPEED = 1
OPPONENT_BRAKE = 0.05
OPPONENT_TOLERANCE = 5.0
OPPONENT_CRUISE = 2.0
OPPONENT_THROTTLE = 0.05
pred_seq: Stim_seq

# class State:
#     def __init__(self):
#         with open("/dev/shm/dishserver_env", "rb") as f:
#             self.memory_map = mmap.mmap(f.fileno(), 0, mmap.MAP_SHARED, mmap.PROT_READ)
#         self.size = len(self.memory_map)
#         mm_ffi = ffi.from_buffer(self.memory_map)
#         self.buffer_in = ffi.cast("struct env_state *", mm_ffi)
#         self.buffer_out = ffi.new("struct env_state *")

#     def read(self):
#         while True:
#             ret = lib.env_stream_atomic_read(self.buffer_in, self.buffer_out, self.size)
#             if ret == 1:
#                 break
#         return self.buffer_out

def get_activity(layout:List[int],
	spike_count:List[int],
	m1:float, m2:float):
    m1 = 0
    m2 = 0
    for i in range(0,MEA_NCHANNEL):
        if layout[i] == MEA_CHANNEL_M1:
            m1 += spike_count[i]
            break
        elif layout[i] == MEA_CHANNEL_M2:
            m2 += spike_count[i]
            break
        else:
            break


def stim_seq_ctrl_check_finished(ctrl:Stim_seq_ctrl, seq_idx:int):
    return ctrl.seq_next_idx[seq_idx] >= ctrl.seq[seq_idx].n_stim

def sawtooth(x , min, max):
    height = max - min
    x-=min
    x= math.fmod(x,2.0 * height)
    x= math.fabs(x)
    if x > height:
        x = 2 * height - x
    x+=min
    return x

def predict_ball_position(config: Pong_config, 
                          state: Pong_state, 
                          frame_no: np.int64, 
                          side: Pong_side):
    top = state.game_area.top.y
    bottom = state.game_area.bottom.y
    right = state.paddle1.position.top.x
    left = state.paddle0.position.bottom.x

    ball = state.ball0
    virtual_prediction = Point(x=0, y=0)
    prediction: Ball_prediction

    if side == PONG_SIDE_LEFT:
        prediction = state.left_prediction
        prediction.position.x = left
        if ball.velocity.x < 0.0:
            virtual_prediction.x = left
        else:
            virtual_prediction.x = 2*right - left
    else:
        assert(side == PONG_SIDE_RIGHT)
        prediction = state.right_prediction
        prediction.position.x = right
        if ball.velocity.x > 0.0:
            virtual_prediction.x = right
        else:
            virtual_prediction.x = 2*left - right
    assert(math.fabs(ball.velocity.x) > 1e-3)
    prediction_time = (virtual_prediction.x - ball.position.x) / ball.velocity.x
    virtual_prediction.y = ball.position.y + prediction_time*ball.velocity.y

    prediction.position.y = sawtooth(virtual_prediction.y,top,bottom)
    prediction.paddle_centre = min(
		max(prediction.position.y, top + config.paddle_size / 2),
		bottom - config.paddle_size / 2)
    prediction.frame_no = frame_no + STATE_STEP_INTERVAL * prediction_time;   

games = []
def end_rally(state: Pong_state,
              frame_no: np.int64,
              elapse: np.int64):
    games.append([state.seed,state.hit_count, state.miss_count, state.rally_length, time.time()])
    pd.DataFrame(games, columns=['seed','hits','miss','rally', 'timestamp']) # .to_csv('RL_Bio_Train.csv')
    # games.append([copy.deepcopy(state), frame_no, elapse])
    state.rally_length = 0
    # state.seed+= 1
    # print("game over ",games[-1][2])
    # print("game over ",games[-1][0].rally_length)

def begin_rally(config: Pong_config, 
                state: Pong_state,
                frame_no: np.int64):
    ret = 0
    paddle_height = config.paddle_size
    state.paddle0.position.top.x = 0
    state.paddle0.position.top.y = AREA_HEIGHT/2.0 - paddle_height/2.0

    state.paddle0.position.bottom.x = PADDLE_WIDTH
    state.paddle0.position.bottom.y = AREA_HEIGHT/2.0 + paddle_height/2.0

    state.paddle0.velocity.x = 0
    state.paddle0.velocity.y = 0

    # Player one paddle setup
    state.paddle1.position.top.x = AREA_WIDTH - PADDLE_WIDTH
    state.paddle1.position.top.y = AREA_HEIGHT/2.0 - paddle_height/2.0

    state.paddle1.position.bottom.x = AREA_WIDTH
    state.paddle1.position.bottom.y = AREA_HEIGHT/2.0 + paddle_height/2.0

    state.paddle1.velocity.x = 0
    state.paddle1.velocity.y = 0

	# Ball setup
    # state.ball0.position.x = random.randrange(AREA_WIDTH)
    # state.ball0.position.y = random.randrange(AREA_HEIGHT)

    state.ball0.position.x = AREA_WIDTH/2.0
    state.ball0.position.y = AREA_HEIGHT/2.0

    state.ball0.velocity.x = 12.0
    # np.random.seed(state.seed) 
    tmp = np.random.uniform()
    #print("random: ",tmp)
    state.ball0.velocity.y = 14.0 * tmp - 2.0

    predict_ball_position(config, state, frame_no, PONG_SIDE_LEFT)
    predict_ball_position(config, state, frame_no, PONG_SIDE_RIGHT)

    state.phase = Phase.NORMAL_PLAY
    state.rally_length = 0

def check_ball_constraints(config:Pong_config, frame_no: np.int64,state: Pong_state):
    ball = state.ball0
    paddle = state.paddle0
    ret = Pong_ball_event.BALL_NO_EVENT

    if ((ball.position.y >= state.game_area.bottom.y) |
         (ball.position.y <=state.game_area.top.y)):
        ball.velocity.y*=-1
        ret = Pong_ball_event.BALL_BOUNCE_WALL
    
    if ball.position.x >= state.paddle1.position.top.x:
        ball.velocity.x*= -1
        predict_ball_position(config, state, frame_no, PONG_SIDE_RIGHT)
        ret = Pong_ball_event.BALL_BOUNCE_OPPONENT
    
    if((ball.position.y >= paddle.position.top.y)&
            (ball.position.y <= paddle.position.bottom.y)&
            (ball.position.x <= paddle.position.bottom.x)):
        ball.velocity.x = math.fabs(ball.velocity.x)
        ball.position.x = paddle.position.bottom.x
        predict_ball_position(config, state, frame_no, PONG_SIDE_LEFT)
        return Pong_ball_event.BALL_BOUNCE_PADDLE

    if ball.position.x <= state.game_area.top.x:
        ball.velocity.x = math.fabs(ball.velocity.x)
        predict_ball_position(config, state, frame_no, PONG_SIDE_LEFT)
        return Pong_ball_event.BALL_MISSED
    
    return ret

def check_paddle_constraints(config: Pong_config, paddle:Paddle, top: float, bottom: float):
    if paddle.position.bottom.y >= bottom:
        paddle.position.bottom.y = bottom
        paddle.position.top.y = bottom - config.paddle_size
        paddle.velocity.y = 0
    
    if paddle.position.top.y <= top:
        paddle.position.top.y = top
        paddle.position.bottom.y = top + config.paddle_size
        paddle.velocity.y = 0    

def stim_seq_ctrl_start(ctrl:Stim_seq_ctrl, idx:int, frame_no:np.int64):
    ctrl.seq_next_idx[idx] = 0
    ctrl.seq_start_tick[idx] = frame_no

def step_ball(ball: Ball):
    ball.position.x += ball.velocity.x
    ball.position.y += ball.velocity.y    

def step_paddle(paddle: Paddle):
	paddle.position.top.x += SPEED*paddle.velocity.x
	paddle.position.bottom.x += SPEED*paddle.velocity.x
	paddle.position.top.y += SPEED*paddle.velocity.y
	paddle.position.bottom.y += SPEED*paddle.velocity.y

def step_mea_paddle(paddle: Paddle, friction:float, paddle_movement:float):
	paddle.velocity.y *= 1.0 - friction
	paddle.velocity.y += paddle_movement
	step_paddle(paddle)

def step_opponent_paddle(paddle:Paddle, target_pos: float):
    diff = target_pos - (paddle.position.top.y + paddle.position.bottom.y)/2
    dir = -1.0 if diff<0 else +1.0
    dist = diff * dir
    vel = paddle.velocity.y * dir
    stopping_dist = 0.5 * vel * vel / OPPONENT_BRAKE
    if vel < 0.00:
        paddle.velocity.y+= OPPONENT_BRAKE * dir
    elif stopping_dist > dist:
        paddle.velocity.y -= OPPONENT_BRAKE * dir
    elif dist < OPPONENT_TOLERANCE:
        paddle.velocity.y = 0.0
    elif vel < OPPONENT_CRUISE:
        paddle.velocity.y+= OPPONENT_THROTTLE * dir
    else:
        paddle.velocity.y = OPPONENT_CRUISE * dir
    step_paddle(paddle)

def pong_state_step(state: Pong_state, config:Main_Config,
                    frame_no:np.int64, 
                    elapse:np.int64,
                    paddle_movement:float,
                    stim_seq_ctrl: Stim_seq_ctrl,
                    stim_freq_ctrl: Stim_freq_ctrl):
    if state.phase == Phase.RANDOM_SEQUENCE:
        if stim_seq_ctrl_check_finished(stim_seq_ctrl, RAND_SEQ_IDX):
            begin_rally(config.env.pong, state, frame_no)
    elif state.phase == Phase.NORMAL_PLAY:
        event = check_ball_constraints(config.env.pong, frame_no, state)
        
        # if event == Pong_ball_event.BALL_NO_EVENT:
        #     print("Ball No Event", frame_no)
        # el
        if event == Pong_ball_event.BALL_MISSED:
            state.miss_count+=1
            #print("Ball missed", frame_no)
            end_rally(state, frame_no, elapse)
        elif event == Pong_ball_event.BALL_BOUNCE_PADDLE:
            predict_ball_position(config.env.pong, state, frame_no, PONG_SIDE_LEFT)
            state.hit_count+=1
            state.rally_length+=1
            #print("ball return", frame_no)
            if config.stim.pred.enabled:
                state.phase = Phase.PRED_SEQUENCE
                stim_seq_ctrl.n_seq = 1
                stim_seq_ctrl.seq[PRED_SEQ_IDX] = pred_seq
                stim_seq_ctrl_start(stim_seq_ctrl, PRED_SEQ_IDX, frame_no)
        elif event == Pong_ball_event.BALL_BOUNCE_OPPONENT:
            predict_ball_position(config.env.pong, state, frame_no, PONG_SIDE_RIGHT)
            #print("ball bounce_opponent", frame_no)
        # elif event == Pong_ball_event.BALL_BOUNCE_WALL:
            #print("ball bounce_wall", frame_no)
        check_paddle_constraints(config.env.pong,
		                         state.paddle0,
		                         state.game_area.top.y,
		                         state.game_area.bottom.y)
        check_paddle_constraints(config.env.pong,
		                         state.paddle1,
		                         state.game_area.top.y,
		                         state.game_area.bottom.y)
        step_mea_paddle(state.paddle0, config.env.pong.friction, paddle_movement)
        step_opponent_paddle(state.paddle1,
		                     state.right_prediction.paddle_centre)
        step_ball(state.ball0)
    else:
        assert(0)
    return state

def init_pong_state(config: Pong_config, state: Pong_state, frame_no: np.int64):
    state.game_area.top.x = 0
    state.game_area.top.y = 0
    state.game_area.bottom.x = AREA_WIDTH
    state.game_area.bottom.y = AREA_HEIGHT
    state.hit_count = 0
    state.miss_count = 0
    begin_rally(config, state, frame_no)

# def env_state_step(state:State, 
#                     config:Config, 
#                     frame_no:np.int64, 
#                     spike_count:List(int),
#                     stim_seq_ctrl: Stim_seq_ctrl,
#                     stim_freq_ctrl):
#     elapse = frame_no - state.fr

#     if elapse < STATE_STEP_INTERVAL:
#         return

#     get_activity()
    
#     if state.phase == Phase.RANDOM_SEQUENCE:
#         return state, config.env,frame_no


game_frame_no = 0
game_config = Pong_config()
game_state = Pong_state()
# game_state = Pong_state(ball0=Ball(position=Point(x=0,y=0), velocity=Point(0,0)),
#                         paddle0=Paddle(position=Rectangle(top = Point(x=0,y=0), bottom=Point(x=0,y=0)),velocity=Point(x=0, y=0)),
#                         paddle1=Paddle(position=Rectangle(top = Point(x=0,y=0), bottom=Point(x=0,y=0)),velocity=Point(x=0, y=0)),
#                         phase=0,
#                         rally_length=0,
#                         left_prediction=Ball_prediction(position=Point(x=0,y=0),paddle_centre=0.0, frame_no=0),
#                         right_prediction= Ball_prediction(position=Point(x=0,y=0),paddle_centre=0.0, frame_no=0),
#                         game_area = Rectangle(top=Point(x=0, y=0),bottom=Point(x=AREA_WIDTH, y=AREA_HEIGHT)),
#                         hit_count= 0, miss_count= 0)
main_config = Main_Config(env=Env_config(open_loop=False, pong=game_config),
                            spike=Spike_config(dummy=0),
                            stim=Stim_config(sensory=Sensory_stim_config(amp=0,phase=0,min_frequency=0,max_frequency=0,enabled=False),
                            random=Random_stim_config(amp=0,phase=0, interval=0,count=0, cooldown=0,pulse_count=0, pulse_interval=0, enabled=False),
                            pred=Pred_stim_config(amp=0,phase=0,pulse_count=0,pulse_interval=0,enabled=False),proprio=Proprio_stim_config(enabled=False),channel=Stim_channel_config(id=[], num=0)),
                            baseline=Baseline_config(dummy=0))

# init_pong_state(game_config, game_state, game_frame_no)
elapse = 0
stim_seq_ctrl = Stim_seq_ctrl(n_seq=0, seq=[],seq_start_tick=[], seq_next_idx=[], last_tick=0)
stim_freq_ctrl = Stim_freq_ctrl(amp=0,phase=0,last_tick=0,spec=[],primary_unit=0)

# print(game_config)

# env_state = State()
# while(1):
#     t1 = time.time()
#     elapse += 2
#     state = env_state.read()
#     a = state.baseline_state.m1_baseline - state.baseline_state.m2_baseline
#     if a==0:
#         paddle_movement = 0
#     elif a > 0:
#         paddle_movement = 2.0
#     elif a < 0:
#         paddle_movement = -2.0
#     else:
#         paddle_movement = 0.0
    
#     pong_state_step(game_state, main_config, state.frame_no, elapse, paddle_movement,stim_seq_ctrl,stim_freq_ctrl)
#     time.sleep(1.0)
#     t2 = time.time()
#     print("\n\n",datetime.now(), (t2 -t1)*1000,game_state)
