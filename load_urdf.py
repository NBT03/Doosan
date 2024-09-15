import pybullet as p
import time
import pybullet_data
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
planeId = p.loadURDF("assets/ur5/plane.urdf")
startPos = [0.33,-0.8,0]
startOrientation = p.getQuaternionFromEuler([0,0,np.pi/2])
# boxId = p.loadURDF("assets/ur5/base_doosan.urdf",startPos, startOrientation)
robot_body_id = p.loadURDF("assets/ur5/doosan_origin.urdf", [0, 0, 0.83], p.getQuaternionFromEuler([0, 0, 0]))
# cabin = p.loadURDF("assets/ur5/Cabin.urdf",[-1.1,0.8,0], p.getQuaternionFromEuler([np.pi/2, 0, 0]))
# tote = p.loadURDF("assets/tote/toteA_large.urdf",[0,0,0], p.getQuaternionFromEuler([np.pi/2, 0, 0]), useFixedBase=True)

robot_joint_info = [p.getJointInfo(robot_body_id, i) for i in range(
            p.getNumJoints(robot_body_id))]
_robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
robot_home_joint_config = [np.pi*-0.5225555555555556,
                           np.pi*-0.07583333333333334,
                           np.pi*0.5659444444444445,
                           np.pi*0.008222222222222223,
                           np.pi * 0.49961111111111123,
                           np.pi * 0.020722222222222222]
_joint_epsilon = 1e-2
def move_joints( target_joint_state, speed=0.1):
    """
        Move robot arm to specified joint configuration by appropriate motor control
    """
    assert len(_robot_joint_indices) == len(target_joint_state)
    p.setJointMotorControlArray(
        robot_body_id, _robot_joint_indices,
        p.POSITION_CONTROL, target_joint_state,
        positionGains=speed * np.ones(len(_robot_joint_indices))
    )

    timeout_t0 = time.time()
    while True:
        # Keep moving until joints reach the target configuration
        current_joint_state = [
            p.getJointState(robot_body_id, i)[0]
            for i in _robot_joint_indices
        ]
        if all([
            np.abs(
                current_joint_state[i] - target_joint_state[i]) < 1e-2
            for i in range(len(_robot_joint_indices))
        ]):
            break
        if time.time() - timeout_t0 > 10:
            print(
                "Timeout: robot is taking longer than 10s to reach the target joint state. Skipping...")
            p.setJointMotorControlArray(
                robot_body_id, _robot_joint_indices,
                p.POSITION_CONTROL, robot_home_joint_config,
                positionGains=np.ones(len(_robot_joint_indices))
            )
            break
            step_simulation(1)
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
while True:
    p.stepSimulation()
    move_joints(robot_home_joint_config,speed=0.01)
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
