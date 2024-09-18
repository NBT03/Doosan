import pybullet as p
import time
import pybullet_data
import numpy as np
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,0)
planeId = p.loadURDF("assets/ur5/plane.urdf")
startPos = [0.75,0.3,0]
startOrientation = p.getQuaternionFromEuler([0,0,np.pi])
boxId = p.loadURDF("assets/ur5/base_doosan.urdf",startPos, startOrientation)
robot_body_id = p.loadURDF("assets/ur5/doosan_origin.urdf", [0, 0, 0.83], p.getQuaternionFromEuler([0, 0, 0]))
cabin = p.loadURDF("assets/ur5/Cabin.urdf",[-0.75,-1,0], p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2]))
tote = p.loadURDF("assets/tote/tote_bin.urdf",[-0.3,-0.35,0.8], p.getQuaternionFromEuler([np.pi/2, 0, 0]), useFixedBase=True)
objec = p.loadURDF("assets/objects/cube.urdf",[-0.1,-0.5,0.84], p.getQuaternionFromEuler([np.pi/2, 0, 0]), useFixedBase=False)


robot_joint_info = [p.getJointInfo(robot_body_id, i) for i in range(
            p.getNumJoints(robot_body_id))]
_robot_joint_indices = [
            x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
while True:
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
