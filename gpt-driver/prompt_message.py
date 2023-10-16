import numpy as np

system_message = """
**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle. Plan a safe 3-second driving trajectory. Avoid collisions with other objects.

Context
- Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. You're at point (0,0).
- Objective: Create a 3-second route using 6 waypoints, one every 0.5 seconds.

Inputs
1. Perception & Prediction: Info about surrounding objects and their predicted movements.
2. Historical Trajectory: Your past 2-second route, given by 4 waypoints.
3. Ego-States: Your current state including velocity, heading angular velocity, can bus data, heading speed, and steering signal.
4. Mission Goal: Goal location for the next 3 seconds.

Task
- Thought Process: Note down critical objects and potential effects from your perceptions and predictions.
- Action Plan: Detail your meta-actions based on your analysis.
- Trajectory Planning: Develop a safe and feasible 3-second route using 6 new waypoints.

Output
- Thoughts:
  - Notable Objects
    Potential Effects
- Meta Action
- Trajectory (MOST IMPORTANT):
  - [(x1,y1), (x2,y2), ... , (x6,y6)]
"""

system_message_cot = """
**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle. Plan a safe 3-second driving trajectory. Avoid collisions with other objects.

Output
- Thoughts: identify critical objects and potential effects from perceptions and predictions.
- Meta Action
- Trajectory (MOST IMPORTANT): 6 waypoints, one every 0.5 seconds
  - [(x1,y1), (x2,y2), ... , (x6,y6)]
"""

system_message_short = """
**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle. Plan a safe 3-second driving trajectory. Avoid collisions with other objects.

Output
- Trajectory (MOST IMPORTANT): 6 waypoints, one every 0.5 seconds
  - [(x1,y1), (x2,y2), ... , (x6,y6)]
"""

def generate_user_message(data, token, perception_range=20.0, short=True):

    # user_message  = f"You have received new input data to help you plan your route.\n"
    user_message  = f"\n"
    
    data_dict = data[token]

    """
    Perception and Prediction Outputs:
        object_boxes: [N, 7]
        object_names: [N]
        object_velocity: [N, 2]
        object_rel_fut_trajs: [N, 12] # diff movements in their local frames
        object_fut_mask: [N, 6]
    """
    object_boxes = data_dict['gt_boxes']
    object_names = data_dict['gt_names']
    # object_velocity = data_dict['gt_velocity']
    object_rel_fut_trajs = data_dict['gt_agent_fut_trajs'].reshape(-1, 6, 2)
    object_fut_trajs = np.cumsum(object_rel_fut_trajs, axis=1) + object_boxes[:, None, :2]
    object_fut_mask = data_dict['gt_agent_fut_masks']
    user_message += f"Perception and Prediction:\n"
    num_objects = object_boxes.shape[0]
    for i in range(num_objects):
        if ((object_fut_trajs[i, :, 1] <= 0).all()) and (object_boxes[i, 1] <= 0): # negative Y, meaning the object is always behind us, we don't care
            continue
        if ((np.abs(object_fut_trajs[i, :, :]) > perception_range).any()) or (np.abs(object_boxes[i, :2]) > perception_range).any(): # filter faraway (> 20m) objects in case there are too many outputs
            continue
        if not short:
            object_name = object_names[i]
            ox, oy = object_boxes[i, :2]
            user_message += f" - {object_name} at ({ox:.2f},{oy:.2f}). "
            user_message += f"Future trajectory: ["
            prediction_ts = 6
            for t in range(prediction_ts):
                if object_fut_mask[i, t] > 0:
                    ox, oy = object_fut_trajs[i, t]
                    user_message += f"({ox:.2f},{oy:.2f})"
                else:
                    ox, oy = "UN", "UN"
                    user_message += f"({ox},{oy})"
                if t != prediction_ts -1:
                    user_message += f", "
            user_message += f"]\n"
        else:
            object_name = object_names[i]
            object_name = object_name.split(".")[-1]
            ox, oy = object_boxes[i, :2]
            user_message += f" - {object_name} at ({ox:.2f},{oy:.2f}), "
            ex, ey = object_fut_trajs[i, -1]
            if object_fut_mask[i, -1] > 0:
                user_message += f"moving to ({ex:.2f},{ey:.2f}).\n"
            else:
                user_message += f"moving to unknown location.\n"

    """
    Ego-States:
        gt_ego_lcf_feat: [vx, vy, ?, ?, v_yaw (rad/s), ego_length, ego_width, v0 (vy from canbus), Kappa (steering)]
    """
    vx = data_dict['gt_ego_lcf_feat'][0]*0.5
    vy = data_dict['gt_ego_lcf_feat'][1]*0.5
    v_yaw = data_dict['gt_ego_lcf_feat'][4]
    ax = data_dict['gt_ego_his_diff'][-1, 0] - data_dict['gt_ego_his_diff'][-2, 0]
    ay = data_dict['gt_ego_his_diff'][-1, 1] - data_dict['gt_ego_his_diff'][-2, 1]
    cx = data_dict['gt_ego_lcf_feat'][2]
    cy = data_dict['gt_ego_lcf_feat'][3]
    vhead = data_dict['gt_ego_lcf_feat'][7]*0.5
    steeling = data_dict['gt_ego_lcf_feat'][8]
    user_message += f"Ego-States:\n"
    user_message += f" - Velocity (vx,vy): ({vx:.2f},{vy:.2f})\n"
    user_message += f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})\n"
    user_message += f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})\n"
    user_message += f" - Can Bus: ({cx:.2f},{cy:.2f})\n"
    user_message += f" - Heading Speed: ({vhead:.2f})\n"
    user_message += f" - Steering: ({steeling:.2f})\n"

    """
    Historical Trjectory:
        gt_ego_his_trajs: [5, 2] last 2 seconds 
        gt_ego_his_diff: [4, 2] last 2 seconds, differential format, viewed as velocity 
    """
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    user_message += f"Historical Trajectory (last 2 seconds):"
    user_message += f" [({xh1:.2f},{yh1:.2f}), ({xh2:.2f},{yh2:.2f}), ({xh3:.2f},{yh3:.2f}), ({xh4:.2f},{yh4:.2f})]\n"
    
    """
    Mission goal:
        gt_ego_fut_cmd
    """
    cmd_vec = data_dict['gt_ego_fut_cmd']
    right, left, forward = cmd_vec
    if right > 0:
        mission_goal = "RIGHT"
    elif left > 0:
        mission_goal = "LEFT"
    else:
        assert forward > 0
        mission_goal = "FORWARD"
    user_message += f"Mission Goal: "
    user_message += f"{mission_goal}\n"
    
    return user_message

def generate_assistant_message(data, token, traj_only = False):

    data_dict = data[token]
    if traj_only:
        assitant_message = ""
    else:
        assitant_message = generate_chain_of_thoughts(data_dict)

    x1 = data_dict['gt_ego_fut_trajs'][1][0]
    x2 = data_dict['gt_ego_fut_trajs'][2][0]
    x3 = data_dict['gt_ego_fut_trajs'][3][0]
    x4 = data_dict['gt_ego_fut_trajs'][4][0]
    x5 = data_dict['gt_ego_fut_trajs'][5][0]
    x6 = data_dict['gt_ego_fut_trajs'][6][0]
    y1 = data_dict['gt_ego_fut_trajs'][1][1]
    y2 = data_dict['gt_ego_fut_trajs'][2][1]
    y3 = data_dict['gt_ego_fut_trajs'][3][1]
    y4 = data_dict['gt_ego_fut_trajs'][4][1]
    y5 = data_dict['gt_ego_fut_trajs'][5][1]
    y6 = data_dict['gt_ego_fut_trajs'][6][1]
    if not traj_only:
        assitant_message += f"Trajectory:\n"    
    assitant_message += f"[({x1:.2f},{y1:.2f}), ({x2:.2f},{y2:.2f}), ({x3:.2f},{y3:.2f}), ({x4:.2f},{y4:.2f}), ({x5:.2f},{y5:.2f}), ({x6:.2f},{y6:.2f})]"
    # assitant_message += f"[ {x1:.2f},{x2:.2f},{x3:.2f},{x4:.2f},{x5:.2f},{x6:.2f},{y1:.2f},{y2:.2f},{y3:.2f},{y4:.2f},{y5:.2f},{y6:.2f} ]"
    return assitant_message

def generate_chain_of_thoughts(data_dict, perception_range=20.0, short=True):
    """
    Generate chain of thoughts reasoning and prompting by simple rules
    """
    ego_fut_trajs = data_dict['gt_ego_fut_trajs']
    ego_his_trajs = data_dict['gt_ego_his_trajs']
    ego_fut_diff = data_dict['gt_ego_fut_diff']
    ego_his_diff = data_dict['gt_ego_his_diff']
    vx = data_dict['gt_ego_lcf_feat'][0]*0.5
    vy = data_dict['gt_ego_lcf_feat'][1]*0.5
    ax = data_dict['gt_ego_his_diff'][-1, 0] - data_dict['gt_ego_his_diff'][-2, 0]
    ay = data_dict['gt_ego_his_diff'][-1, 1] - data_dict['gt_ego_his_diff'][-2, 1]
    ego_estimate_velos = [
        [0, 0],
        [vx, vy],
        [vx+ax, vy+ay],
        [vx+2*ax, vy+2*ay],
        [vx+3*ax, vy+3*ay],
        [vx+4*ax, vy+4*ay],
        [vx+5*ax, vy+5*ay],
    ]
    ego_estimate_trajs = np.cumsum(ego_estimate_velos, axis=0) # [7, 2]
    # print(ego_estimate_trajs)
    object_boxes = data_dict['gt_boxes']
    object_names = data_dict['gt_names']
    
    object_rel_fut_trajs = data_dict['gt_agent_fut_trajs'].reshape(-1, 6, 2)
    object_fut_trajs = np.cumsum(object_rel_fut_trajs, axis=1) + object_boxes[:, None, :2]
    object_fut_trajs = np.concatenate([object_boxes[:, None, :2], object_fut_trajs], axis=1)
    object_fut_mask = data_dict['gt_agent_fut_masks']
    num_objects = object_boxes.shape[0]

    num_future_horizon = 7 # include current
    object_collisons = np.zeros((num_objects, num_future_horizon))
    for i in range(num_objects):
        if (object_fut_trajs[i, :, 1] <= 0).all(): # negative Y, meaning the object is always behind us, we don't care
            continue
        if (np.abs(object_fut_trajs[i, :, :]) > perception_range).any(): # filter faraway (> 20m) objects in case there are too many outputs
            continue
        for t in range(num_future_horizon):
            mask = object_fut_mask[i, t-1] > 0 if t > 0 else True
            if not mask: continue
            ego_x, ego_y = ego_estimate_trajs[t]
            object_x, object_y = object_fut_trajs[i, t]
            size_x, size_y = object_boxes[i, 3:5] * 0.5 # half size
            collision = collision_detection(ego_x, ego_y, 0.925, 2.04, object_x, object_y, size_x, size_y)
            if collision:
                object_collisons[i, t] = 1
                # import pdb; pdb.set_trace()    
                break

    assitant_message = f"Thoughts:\n"
    if (object_collisons==0).all(): # nothing to care about
        assitant_message += f" - Notable Objects from Perception: None\n"
        assitant_message += f"   Potential Effects from Prediction: None\n"
        # assitant_message += f"   Nothing to care.\n"
    else:
        for i in range(num_objects):
            for t in range(num_future_horizon):
                if object_collisons[i, t] > 0:
                    object_name = object_names[i]
                    if short:
                        object_name = object_name.split(".")[-1]
                    ox, oy = object_boxes[i, :2]
                    time = t*0.5
                    # assitant_message += f" ################################################################################\n"
                    assitant_message += f" - Notable Objects from Perception: {object_name} at ({ox:.2f},{oy:.2f})\n"
                    assitant_message += f"   Potential Effects from Prediction: within the safe zone of the ego-vehicle at the {time}-second timestep\n"
    meta_action = generate_meta_action(
        ego_fut_diff=ego_fut_diff, 
        ego_fut_trajs=ego_fut_trajs, 
        ego_his_diff=ego_his_diff, 
        ego_his_trajs=ego_his_trajs
    )
    assitant_message += ("Meta Action: " + meta_action)
    return assitant_message

def collision_detection(x1, y1, sx1, sy1, x2, y2, sx2, sy2, x_space=1.0, y_space=3.0): # safe distance
    if (np.abs(x1-x2) < sx1+sx2+x_space) and (y2 > y1) and (y2 - y1 < sy1+sy2+y_space): # in front of you
        return True
    else:
        return False

def generate_meta_action( 
    ego_fut_diff,
    ego_fut_trajs,
    ego_his_diff,
    ego_his_trajs,
    ):
    meta_action = ""

    # speed meta
    constant_eps = 0.5
    his_velos = np.linalg.norm(ego_his_diff, axis=1)
    fut_velos = np.linalg.norm(ego_fut_diff, axis=1)
    cur_velo = his_velos[-1]
    end_velo = fut_velos[-1]

    if cur_velo < constant_eps and end_velo < constant_eps:
        speed_meta = "stop"
    elif end_velo < constant_eps:
        speed_meta = "a deceleration to zero"
    elif np.abs(end_velo - cur_velo) < constant_eps:
        speed_meta = "a constant speed"
    else:
        if cur_velo > end_velo:
            if cur_velo > 2 * end_velo:
                speed_meta = "a quick deceleration"
            else:
                speed_meta = "a deceleration"
        else:
            if end_velo > 2 * cur_velo:
                speed_meta = "a quick acceleration"
            else:
                speed_meta = "an acceleration"
    
    # behavior_meta
    if speed_meta == "stop":
        meta_action += (speed_meta + "\n")
        return meta_action.upper()
    else:
        forward_th = 2.0
        lane_changing_th = 4.0
        if (np.abs(ego_fut_trajs[:, 0]) < forward_th).all():
            behavior_meta = "move forward"
        else:
            if ego_fut_trajs[-1, 0] < 0: # left
                if np.abs(ego_fut_trajs[-1, 0]) > lane_changing_th:
                    behavior_meta = "turn left"
                else:
                    behavior_meta = "chane lane to left"
            elif ego_fut_trajs[-1, 0] > 0: # right
                if np.abs(ego_fut_trajs[-1, 0]) > lane_changing_th:
                    behavior_meta = "turn right"
                else:
                    behavior_meta = "change lane to right"
            else:
                raise ValueError(f"Undefined behaviors: {ego_fut_trajs}")
        
        # heading-based rules
        # ego_fut_headings = np.arctan(ego_fut_diff[:,0]/(ego_fut_diff[:,1]+1e-4))*180/np.pi # in degree
        # ego_his_headings = np.arctan(ego_his_diff[:,0]/(ego_his_diff[:,1]+1e-4))*180/np.pi # in degree
        
        # forward_heading_th = 5 # forward heading is always near 0
        # turn_heading_th = 45

        # if (np.abs(ego_fut_headings) < forward_heading_th).all():
        #     behavior_meta = "move forward"
        # else:
        #     # we extract a 5-s curve, if the largest heading change is above 45 degrees, we view it as turn
        #     curve_headings = np.concatenate([ego_his_headings, ego_fut_headings])
        #     min_heading, max_heading = curve_headings.min(), curve_headings.max()
        #     if ego_fut_trajs[-1, 0] < 0: # left
        #         if np.abs(max_heading - min_heading) > turn_heading_th:
        #             behavior_meta = "turn left"
        #         else:
        #             behavior_meta = "chane lane to left"
        #     elif ego_fut_trajs[-1, 0] > 0: # right
        #         if np.abs(max_heading - min_heading) > turn_heading_th:
        #             behavior_meta = "turn right"
        #         else:
        #             behavior_meta = "chane lane to right"
        #     else:
        #         raise ValueError(f"Undefined behaviors: {ego_fut_trajs}")
        
        meta_action += (behavior_meta + " with " + speed_meta + "\n")
        return meta_action.upper()


# system_message = """
# As a professional autonomous driving system, you are tasked with plotting a secure and human-like path within a 3-second window using the following guidelines and inputs:

# ### Context
# - **Coordinate System**: You are in the ego-vehicle coordinate system positioned at (0,0). The X-axis is perpendicular to your heading direction, while the Y-axis represents the heading direction.
# - **Location**: You are mounted at the center of an ego-vehicle that has 4.08 meters length and 1.85 meters width.
# - **Objective**: Generate a route characterized by 6 waypoints, with a new waypoint established every 0.5 seconds.

# ### Inputs
# 1. **Perception & Prediction** (You observe the surrounding objects and estimate their future movements):
#    - object name at (ox1, ox2). Future trajectory: [(oxt1, oyt1), ..., (oxt6, oyt6)], 6 waypoints in 3 seconds, UN denotes future location at that timestep is unknown
#    - ...
   
# 2. **Historical Trajectory** (Your historital trajectory from the last 2 seconds, presented as 4 waypoints):
#    - [(xh1, yh1), (xh2, yh2), (xh3, yh3), (xh4, yh4)]

# 3. **Ego-States** (Your current states):
#    - **Velocity** (vx, vy) # meters per 0.5 second
#    - **Heading Angular Velocity** (v_yaw) # ego-vehicle heading change rate, rad per second
#    - **Acceleration** (ax, ay) # velocity change rate per 0.5 second
#    - **Heading Speed** # meters per 0.5 second
#    - **Steering** # steering signal
   
# 4. **Mission Goal**: Instructions outlining your objectives for the upcoming 3 seconds.

# ### Task
# - Integrate and process all the above inputs to construct a driving route.
# - Thinking about what you have received and make driving decisions. Write down your thoughts and the action. 
# - Output a set of 6 new waypoints for the upcoming 3 seconds (Note: This task is of the most importance!). These should be formatted as coordinate pairs:
#    - (x1, y1) # 0.5 second
#    - (x2, y2) # 1.0 second
#    - (x3, y3) # 1.5 second
#    - (x4, y4) # 2.0 second
#    - (x5, y5) # 2.5 second
#    - (x6, y6) # 3.0 second
# - Final output format: 
#     Thoughts:
#     - Notable Objects from Perception: ...
#       Potential Effects from Prediction: ...
#     Meta Action:
#       ...
#     Trajectory:
#     - [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]

# Ensure the safety and feasibility of the path devised within the given 3-second timeframe. Let's work on crafting a safe route!
# """

def generate_incontext_message(data, token):
    incontext_message = "\nFor example:\n"
    incontext_message += "Input:\n"
    user_message = generate_user_message(data, token)
    incontext_message += user_message
    incontext_message += "You should generate the following content:\n"
    assistant_message = generate_assistant_message(data, token)
    incontext_message += assistant_message
    return incontext_message