import pybullet as p
import pybullet_data
import numpy as np
import time
import os 
import imageio
import matplotlib.pyplot as plt

from rx150.rx150_env import RX150Env2
from stable_baselines3 import SAC


'''
in console (if using lightning.ai) to forward the display to browser : 

Xvfb :99 -screen 0 1920x1080x24 & export DISPLAY=:99
x11vnc -display :99 -rfbport 5901 -shared -forever &
websockify --web=/usr/share/novnc 8090 localhost:5901 
 
'''

'''
    Basic code to view the trained SAC
    Press 'R' to reset the environment
'''

def read_parameters(dbg_params):
    '''Reads values from debug parameters
    
    Parameters
    ----------
    dbg_params : dict
        Dictionary where the keys are names (str) of parameters and the values are
        the itemUniqueId (int) for the corresponing debug item in pybullet
    
    Returns
    -------
    dict 
        Dictionary that maps parameter names (str) to parameter values (float)
    '''
    values = dict()
    for name, param in dbg_params.items():
        values[name] = p.readUserDebugParameter(param)

    return values

def interactive_camera_placement(pos_scale=1.,
                                 max_dist=10.,
                                 show_plot=True,
                                 verbose=True,
                                 ):
    '''GUI for adjusting camera placement in pybullet. Use the scales to adjust
    intuitive parameters that govern view and projection matrix.  When you are
    satisfied, you can hit the print button and the values needed to recreate
    the camera placement will be logged to console.
    In addition to showing a live feed of the camera, there are also two visual
    aids placed in the simulator to help understand camera placement: the target
    position and the camera. These are both shown as red objects and can be
    viewed using the standard controls provided by the GUI.
    Note
    ----
    There must be a simulator running in GUI mode for this to work
    Parameters
    ----------
    pos_scale : float
        Position scaling that limits the target position of the camera.
    max_dist : float
        Maximum distance the camera can be away from the target position, you
        may need to adjust if you scene is large
    show_plot : bool, default to True
        If True, then a matplotlib window will be used to plot the generated
        image.  This is beneficial if you want different values for image width
        and height (since the built in image visualizer in pybullet is always
        square).
    verbose : bool, default to False
        If True, then additional parameters will be printed when print button
        is pressed.
    '''
    np.set_printoptions(suppress=True, precision=4)

    dbg = dict()
    # for view matrix
    dbg['target_x'] = p.addUserDebugParameter('target_x', -pos_scale, pos_scale, 0)
    dbg['target_y'] = p.addUserDebugParameter('target_y', -pos_scale, pos_scale, 0)
    dbg['target_z'] = p.addUserDebugParameter('target_z', -pos_scale, pos_scale, 0)
    dbg['distance'] = p.addUserDebugParameter('distance', 0, max_dist, max_dist/2)
    dbg['yaw'] =  p.addUserDebugParameter('yaw', -180, 180, 0)
    dbg['pitch'] =  p.addUserDebugParameter('pitch', -180, 180, -40)
    dbg['roll'] =  p.addUserDebugParameter('roll', -180, 180, 0)
    dbg['upAxisIndex'] =  p.addUserDebugParameter('toggle upAxisIndex', 1, 0, 1)

    # for projection matrix
    dbg['width'] = p.addUserDebugParameter('width', 64, 1000, 320)
    dbg['height'] = p.addUserDebugParameter('height', 64, 1000, 240)
    dbg['fov'] = p.addUserDebugParameter('fov', 1, 180, 50)
    dbg['near_val'] = p.addUserDebugParameter('near_val', 1e-6, 1, 0.1)
    dbg['far_val'] = p.addUserDebugParameter('far_val', 1, 100, 10)

    # visual aids for target and camera pose
    target_vis_id = p.createVisualShape(p.GEOM_SPHERE,
                                         radius=0.01,
                                         rgbaColor=[1,0,0,0.7])
    target_body = p.createMultiBody(0, -1, target_vis_id)

    camera_vis_id = p.createVisualShape(p.GEOM_BOX,
                                         halfExtents=[0.02, 0.05, 0.02],
                                         rgbaColor=[1,0,0,0.7])
    camera_body = p.createMultiBody(0, -1, camera_vis_id)

    # pyplot window to show feed
    if show_plot:
        plt.figure()
        plt_im = plt.imshow(np.zeros((240,320,4)))
        plt.axis('off')
        plt.tight_layout(pad=0)

    dbg['print'] =  p.addUserDebugParameter('print params', 1, 0, 1)
    old_print_val = 1
    while 1:
        values = read_parameters(dbg)

        target_pos = np.array([values[f'target_{c}'] for c in 'xyz'])
        upAxisIndex = (int(values['upAxisIndex']) % 2 ) + 1
        view_mtx = p.computeViewMatrixFromYawPitchRoll(target_pos,
                                                        values['distance'],
                                                        values['yaw'],
                                                        values['pitch'],
                                                        values['roll'],
                                                        upAxisIndex)

        width = int(values['width'])
        height = int(values['height'])
        aspect = width/height
        proj_mtx = p.computeProjectionMatrixFOV(values['fov'],
                                                 aspect,
                                                 values['near_val'],
                                                 values['far_val'])

        #update visual aids for camera, target pos
        p.resetBasePositionAndOrientation(target_body, target_pos, [0,0,0,1])

        view_mtx = np.array(view_mtx).reshape((4,4),order='F')
        cam_pos = np.dot(view_mtx[:3,:3].T, -view_mtx[:3,3])
        cam_euler = np.radians([values['pitch'],values['roll'],values['yaw']])
        cam_quat = p.getQuaternionFromEuler(cam_euler)
        p.resetBasePositionAndOrientation(camera_body, cam_pos, cam_quat)

        view_mtx = view_mtx.reshape(-1, order='F')
        img = p.getCameraImage(width, height, view_mtx, proj_mtx)[2]
        if show_plot:
            plt_im.set_array(img)
            plt.gca().set_aspect(height/width)
            plt.draw()
            plt.pause(0.1)

        if old_print_val != values['print']:
            old_print_val = values['print']
            print("\n========================================")
            print(f"VIEW MATRIX : \n{np.array(view_mtx)}")
            print(f"PROJECTION MATRIX : \n{np.array_str(view_mtx)}")
            if verbose:
                print(f"target position : {np.array_str(target_pos)}")
                print(f"distance : {dbg['distance']:.2f}")
                print(f"yaw : {dbg['yaw']:.2f}")
                print(f"pitch : {dbg['pitch']:.2f}")
                print(f"roll : {dbg['roll']:.2f}")
                print(f"upAxisIndex : {upAxisIndex:d}")
                print(f"width : {width:d}")
                print(f"height : {height:d}")
                print(f"fov : {dbg['fov']:.1f}")
                print(f"aspect : {aspect:.2f}")
                print(f"nearVal : {dbg['near_val']:.2f}")
                print(f"farVal : {dbg['far_val']:.2f}")
            print("========================================\n")


urdf_path = "/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/rx150.urdf"

total_it = 0

rx_env = RX150Env2(
    urdf_path=urdf_path, 
    headless=False,
    max_timesteps=1_000_000,
    image_only=False,
    task='stack',
    use_intrinsic=True,
)

rx_env.reset()

# Trained SAC
# model = SAC.load("/teamspace/studios/this_studio/IFT6163-Group4/trained_models/SAC_sparse_trained")
model = SAC.load("/teamspace/studios/this_studio/IFT6163-Group4/SAC_rx150_preference_40")

# Untrained model
# model = SAC("MultiInputPolicy", rx_env, verbose=1)
# model = SAC("CnnPolicy", rx_env, verbose=1)

# interactive_camera_placement()

while True : 

    keys = p.getKeyboardEvents()
    if (ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED) or total_it % 1000 == 0:
        print("Resetting env")
        rx_env.reset()

    total_it += 1

    ob = rx_env.get_obs()
    # print(ob)
    action = model.predict(ob)[0]
    print(action)

    _,rw,done,_,_ = rx_env.step(action)

    # if total_it % 10 == 0 :
        # print(f"reward : {rw}")
        # print(f"reward : {rw}, sqr dist : {((np.array(rx_env.get_end_effector_pos()) - rx_env.target_pos)**2).sum()}")

    if done : 
        print("TASK ACCOMPLISHED")
        break

    time.sleep(0.005)
        
rx_env.close()

