
import rclpy
import numpy as np
from GazeboRX150Env.gazebo_rx150_env import GazeboRX150Env
import time

def main():
    rclpy.init()
    env = GazeboRX150Env()    
    obs, _ = env.reset()
    print("Env ready. Press <Enter> to send a grasp...")
    for _ in range(10):
        rclpy.spin_once(env.node, timeout_sec=0.1)
        time.sleep(0.5)

    while True:
        input()

        #  Wait for depth to be available 
        while env.current_depth is None :
            rclpy.spin_once(env.node, timeout_sec=0.1)
        #env.grasp_net._cam_info(env.cam_model)
        # Predict grasp in camera frame 
        grasp_cam = env.grasp_net.predict(
            env.current_image, env.current_depth, "red object "
        )
        print(f"grasp (camera frame): {grasp_cam}")

        # Transform grasp to world frame 
        grasp_world = env.transform_camera_to_world(grasp_cam)
        print(f" grasp (world frame): {grasp_world}")

        # joint targets and publish
        joints = env._pose_to_joints(grasp_world)
        print(f" joint target: {np.round(joints, 3)}")

        env._publish_arm(joints)

        #  Let simulation update 
        rclpy.spin_once(env.node, timeout_sec=0.5)
        eef = env._eef_pose()
        print(f" new EEF pose: {np.round(eef, 4)}")

        d = np.linalg.norm(eef[:3] - grasp_world[:3])
        print(f" distance to grasp: {d:.4f} m")

        #  Retry or quit 
        if input("\nPress <q> to quit, <Enter> to try again: ").lower() == "q":
            break

    env.close()

if __name__ == "__main__":
    main()
