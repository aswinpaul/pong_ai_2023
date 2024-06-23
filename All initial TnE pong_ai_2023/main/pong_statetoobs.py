def bx_obs(bx):
    
    # x-position of ball
    cond = True
    bx_ranges = [[-100,15]]
    x = 15
    while(cond):
        x2 = x + 15
        bx_ranges.append([x,x2])
        x = x2
        if(x2 > 600):
            cond = False
    for r in bx_ranges:
        if bx in range(r[0], r[1]):
            o1 = bx_ranges.index(r)
            return(o1)

def by_obs(by):
    
    # y-position of ball
    cond = True
    by_ranges = [[-100,75]]
    x = 75
    while(cond):
        x2 = x + 75
        by_ranges.append([x,x2])
        x = x2
        if(x2 > 600):
            cond = False
    for r in by_ranges:
        if by in range(r[0], r[1]):
            o2 = by_ranges.index(r)
            return(o2)

def state_to_obs(state):
    
        bx = int(state[0])
        by = int(state[1])
        pp = int((state[3]+state[4])/2)
    
        o1 = bx_obs(bx)
        o2 = by_obs(by)
        o3 = by_obs(pp)
        
        return([o1, o2, o3])