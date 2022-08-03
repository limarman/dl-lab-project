# episode count
def init_global_episode():
    global episode_number
    episode_number = 0

def return_episode_number():
    return episode_number

def increment_episode_number():
    global episode_number
    episode_number += 1

# step count
def init_global_step():
    global step_count
    step_count = 0

def return_step_count():
    return step_count

def increment_step_count():
    global step_count
    step_count += 1
