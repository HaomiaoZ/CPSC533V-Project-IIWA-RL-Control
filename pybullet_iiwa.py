import pybullet as p
import time
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)

goal_visualization_shape_id = p.createVisualShape(p.GEOM_CAPSULE, radius =0.1, length=0.2, rgbaColor =[0,1,0,0.4], specularColor=[1, 1, 1])
goal_visualization_id =p.createMultiBody(baseVisualShapeIndex=goal_visualization_shape_id)

#planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])
iiwaId = p.loadURDF("kuka_iiwa/model.urdf",startPos, startOrientation,useFixedBase=True)# there are other configurations under this directory as well

#set the center of mass frame (loadURDF sets base link frame)
#startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos,
#startOrientation)

p.resetBasePositionAndOrientation(iiwaId, startPos, startOrientation)
p.resetBasePositionAndOrientation(goal_visualization_id, [0,0,1], startOrientation)

# get joint limit
joint_num = p.getNumJoints(iiwaId)

joint_limit =np.zeros((joint_num,2))

# getting lower and upper joint limit
for i in range(joint_num):
    joint_info = p.getJointInfo(iiwaId,i)
    joint_limit[i]= joint_info[8:10]

target_joint_pos = np.random.uniform(joint_limit[:,0],joint_limit[:,1])
for ii in range(10):

    #find target position in joint space to ensure valid end effector position, find end effector position
    target_joint_pos = np.random.uniform(joint_limit[:,0],joint_limit[:,1])
    for num in range(joint_num):
        p.resetJointState(iiwaId, num, target_joint_pos[num])
    target_eef_state = p.getLinkState(iiwaId,6)
    p.resetBasePositionAndOrientation(goal_visualization_id, target_eef_state[4], target_eef_state[5])

    #randomly initialize to a valid state
    initial_joint_pos = np.random.uniform(joint_limit[:,0],joint_limit[:,1])
    for num in range(joint_num):
        p.resetJointState(iiwaId, num, initial_joint_pos[num])
    
    for i in range (1000):
        p.setJointMotorControlArray(iiwaId,jointIndices=[i for i in range(joint_num)], controlMode=p.POSITION_CONTROL, targetPositions = target_joint_pos)
        
        p.stepSimulation()

        joint_states= p.getJointStates(iiwaId,jointIndices=[i for i in range(joint_num)])
        state_vec = np.array([joint_state[0:2] for joint_state in joint_states]).flatten()

        time.sleep(1./240.)

Pos, Orn = p.getBasePositionAndOrientation(iiwaId)

print(Pos,Orn)
p.disconnect()