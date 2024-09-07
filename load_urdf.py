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
boxId = p.loadURDF("assets/ur5/base_doosan.urdf",startPos, startOrientation)
robotid = p.loadURDF("assets/ur5/doosan_origin.urdf", [0, 0, 0.83], p.getQuaternionFromEuler([0, 0, 0]))
cabin = p.loadURDF("assets/ur5/Cabin.urdf",[-1.1,0.8,0], p.getQuaternionFromEuler([np.pi/2, 0, 0]))
tote = p.loadURDF("assets/tote/toteA_large.urdf",[0,0,0], p.getQuaternionFromEuler([np.pi/2, 0, 0]), useFixedBase=True)

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
while True:
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()
