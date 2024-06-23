def pong_state_to_obs(state, reward, factor):
    
    ball_x = state[0]
    o1_float = ball_x/factor
    o1 = int(o1_float)

    ball_y = state[1] 
    o2_float = ball_y/factor
    o2 = int(o2_float)

    ball_vx = state[2]
    o3_float = ball_vx
    o3_i = int(o3_float)
    o3 = 0 if o3_i==2 else 1 


    ball_vy = state[3]
    o4_float = ball_vx
    o4_i = int(o4_float)
    o4 = 0 if o4_i==2 else 1 

    paddle_pos = state[4]
    o5_float = paddle_pos/factor
    o5 = int(o5_float)

    paddle_vel = state[5]
    o6_float = paddle_vel
    o6_i = int(o6_float)
    o6 = 0 if o6_i==4 else 1 
    
    if(reward == -1):
        o7 = 0
    elif(reward == 1):
        o7 = 2
    else:
        o7 = 1
    
    observation = [o1, o2, o3, o4, o5, o6, o7]
    
    return(observation)