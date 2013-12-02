#! /usr/bin/env python


import cProfile
import numpy as np
import pstats
import argparse

"""
Run a BECCA agent with a world.

To use this module as a top level script, select the World that the Agent 
will be placed in.
Make sure the appropriate import line is included and uncommented below. 
Run from the command line, e.g. 
> python tester.py
"""

# Worlds from the benchmark
#from worlds.base_world import World
#from worlds.grid_1D import World
#from worlds.grid_1D_ms import World
#from worlds.grid_1D_noise import World
#from worlds.grid_2D import World
#from worlds.grid_2D_dc import World
#from worlds.image_1D import World
#from worlds.image_2D import World

# If you want to run a world of your own, add the appropriate line here

from core.agent import Agent 

testing_lifespan = 10 ** 8
profiling_lifespan = 10 ** 4

def test(world, restore=False, show=True, agent_name=None):
    """ 
    Run BECCA with world.  
    
    If restore=True, this method loads a saved agent if it can find one.
    Otherwise it creates a new one. It connects the agent and
    the world together and runs them for as long as the 
    world dictates.
    
    To profile BECCA's performance with world, manually set
    profile_flag in the top level script environment to True.
    """
    if agent_name is None:
        agent_name = '_'.join((world.name, 'agent'))
    agent = Agent(world.num_sensors, world.num_actions, 
                  agent_name=agent_name, show=show)
    if restore:
        agent = agent.restore()

    # If configured to do so, the world sets some BECCA parameters to 
    # modify its behavior. This is a development hack, and 
    # should eventually be removed as BECCA matures and settles on 
    # a good, general purpose set of parameters.
    world.set_agent_parameters(agent)
    actions = np.zeros((world.num_actions,1))
    
    # Repeat the loop through the duration of the existence of the world 
    while(world.is_alive()):
        sensors, reward = world.step(actions)
        actions = agent.step(sensors, reward)
        world.visualize(agent)
    return agent.report_performance()

def profile():
    """ Profile BECCA's performance """
    cProfile.run('test(World(lifespan=profiling lifespan), restore=True)', 
                 'tester_profile')
    p = pstats.Stats('tester_profile')
    p.strip_dirs().sort_stats('time', 'cum').print_stats(30)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BECCA against possible worlds')
    parser.add_argument("-w", "--world", help="choose which world to run", default="watch")
    parser.add_argument("-t", "--test", help="Enable testing mode")
    parser.add_argument("-p", "--profile", help="Begin Profiling mode")
    parser.add_argument("--viz", type=int, help="Visualization period (in timesteps)", default=10**4)
    parser.add_argument("-o", "--horizontal", type=int, help="Horizontal size (in pixels)", default=40)
    parser.add_argument("-v", "--vertical", type=int, help="Vertical size (in pixels)", default=40)
    parser.add_argument("--testlife", type=int, help="Testing lifespan", default=10**8)
    parser.add_argument("--profilelife", type=int, help="Profiling lifespan", default=10**4)
    args = parser.parse_args()
    

    if args.world == "listen":
        from becca_world_listen.listen import World
        if args.profile:
            profile()
        else:
            test(World(lifespan=args.testlife, test=args.test,visualize_period=args.viz), restore=True)
    elif args.world == "watch":
        from becca_world_watch.watch import World
        if args.profile:
            profile()
        else:
            test(World(lifespan=args.testlife, test=args.test, fov_horz_span=args.horizontal, fov_vert_span=args.vertical, visualize_period=args.viz), restore=True)
    elif args.world == "tiny_images":
        from becca_world_tiny_images.tiny_images import World
        if args.profile:
            profile()
        else:
            test(World(lifespan=args.testlife, test=args.test, visualize_period=args.viz), restore=True)
    elif args.world == "audio_video":
        from becca_world_audio_video.audio_video import World
        if args.profile:
            profile()
        else:
            test(World(lifespan=args.testlife, test=args.test, fov_horz_span=args.horizontal, fov_vert_span=args.vertical, visualize_period=args.viz), restore=True)