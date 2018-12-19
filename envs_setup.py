import marlo
from marlo import experiments

import gym
from gym import spaces
import gym.wrappers


def start(gamename, width=256, height=192):
    global client_width
    global client_height
    global obs_size
    client_width = width
    client_height = height
    obs_size = client_width * client_height

    client_pool = [('127.0.0.1', 10000)]
    envparams = {"client_pool": client_pool,
                 "videoResolution": [client_width, client_height]
                 }

    if gamename == 'MarLo-Attic-v0':
        """
        Actions:
                'move 0\nturn 0',
                'jump 1', 'jump 0',
                'move 1', 'move -1',
                'pitch 1', 'pitch -1',
                'strafe 1', 'strafe -1',
                'turn 1', 'turn -1',
                'crouch 1', 'crouch 0',
                'use 1', 'use 0'
        """
        action_select = list(range(15))

    elif gamename == 'MarLo-FindTheGoal-v0':
        """
        Actions:
                'move 0\nturn 0',
                'jump 1', 'jump 0',
                'move 1', 'move -1',
                'pitch 1', 'pitch -1',
                'strafe 1', 'strafe -1',
                'turn 1', 'turn -1',
                'crouch 1', 'crouch 0',
                'use 1', 'use 0'
        """
        action_select = [0, 3, 4, 5, 6, 9, 10, 13, 14]

    elif gamename == 'MarLo-Vertical-v0':
        """
        Actions:
                'move 0\nturn 0',
                'jump 1', 'jump 0',
                'move 1', 'move -1',
                'pitch 1', 'pitch -1',
                'strafe 1', 'strafe -1',
                'turn 1', 'turn -1',
                'crouch 1', 'crouch 0',
                'use 1', 'use 0'
        """
        action_select = [0, 3, 4, 9, 10, 13, 14]

    elif gamename == 'MarLo-TrickyArena-v0':
        """
        Actions:
                'move 0\nturn 0',
                'jump 1', 'jump 0',
                'move 1', 'move -1',
                'pitch 1', 'pitch -1',
                'strafe 1', 'strafe -1',
                'turn 1', 'turn -1',
                'crouch 1', 'crouch 0',
                'use 1', 'use 0'
        """
        action_select = [0, 3, 9, 10]

    elif gamename == 'MarLo-Obstacles-v0':
        """
        Actions:
                'move 0\nturn 0',
                'jump 1', 'jump 0',
                'move 1', 'move -1',
                'pitch 1', 'pitch -1',
                'strafe 1', 'strafe -1',
                'turn 1', 'turn -1',
                'crouch 1', 'crouch 0',
                'use 1', 'use 0'
        """
        action_select = list(range(15))

    elif gamename == 'MarLo-MazeRunner-v0':
        """
        Actions:
                'move 0\nturn 0',
                'jump 1', 'jump 0',
                'move 1', 'move -1',
                'pitch 1', 'pitch -1',
                'turn 1', 'turn -1',
                'crouch 1', 'crouch 0',
                'attack 1', 'attack 0',
                'use 1', 'use 0'
        """
        action_select = [0, 1, 2, 3, 7, 8]

    elif gamename == 'MarLo-CliffWalking-v0':
        """
        Actions:
                'move 0\nturn 0',
                'move 1', 'move -1',
                'jumpmove 1', 'jumpmove -1',
                'strafe 1', 'strafe -1',
                'jumpstrafe 1', 'jumpstrafe -1',
                'turn 1', 'turn -1',
                'movenorth 1', 'moveeast 1', 'movesouth 1', 'movewest 1',
                'jumpnorth 1', 'jumpeast 1', 'jumpsouth 1', 'jumpwest 1',
                'jump 1',
                'look 1', 'look -1',
                'use 1',
                'jumpuse 1'
        """
        action_select = [0, 1, 9, 10]
    elif gamename == 'MarLo-CatchTheMob-v0':
        """
        'move 0\nturn 0',
        'move 1', 'move -1',
        'jumpmove 1', 'jumpmove -1',
        'strafe 1', 'strafe -1',
        'jumpstrafe 1', 'jumpstrafe -1',
        'turn 1', 'turn -1',
        'movenorth 1', 'moveeast 1', 'movesouth 1', 'movewest 1',
        'jumpnorth 1', 'jumpeast 1', 'jumpsouth 1', 'jumpwest 1',
        'jump 1',
        'look 1','look -1',
        'use 1',
        'jumpuse 1'
        """
        action_select = [0, 1, 9, 10, 22]
    elif gamename == 'MarLo-Eating-v0':
        """
        Actions:
                'move 0\nturn 0',
                'jump 1','jump 0',
                'move 1','move -1',
                'pitch 1','pitch -1',
                'strafe 1','strafe -1',
                'turn 1','turn -1',
                'crouch 1', 'crouch 0',
                'use 1', 'use 0']
        """
        action_select = [0, 3, 9, 10, ]
        envparams = {"client_pool": client_pool,
                     "videoResolution": [320, 240],
                     "observeHotBar": True
                     }
    else:
        action_select = [0, 1, 2, 3, 7, 8]  # default
    n_actions = len(action_select)
    action_space = list(range(n_actions))

    # Ensure that you have a minecraft-client running with : marlo-server
    # --port 10000
    join_tokens = marlo.make(gamename, params=envparams)

    assert len(join_tokens) == 1
    join_token = join_tokens[0]

    global env
    env = marlo.init(join_token)

    return n_actions, action_space, action_select
