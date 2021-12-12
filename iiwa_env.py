import pybullet as p
import time
import pybullet_data
import numpy as np

class IIWAEnv():
    '''
    Description
    A IIWA Arm to reach a spocific position and orientation with positon and torque control

    Observation
    IIWA Arm 7 DOF joint position and velocity, target end effector position and orientation (in quaternion for now)

    Action
    7 DOF position control for now (want to add torque control)

    Reward
    position error and orientation error

    starting state
    all 7 joint angles are started with uniform ditribution within joint limit


    Termination
    After 2400 step (10s since pybullet is running in 240Hz), or the pose error is smaller than 1e-4

    '''

    #TODO add control mode
    def __init__(self):
        #connect to pybullet
        physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-9.81)

        # create goal visualization
        goal_visualization_shape_id = p.createVisualShape(p.GEOM_CAPSULE, radius =0.1, length=0.2, rgbaColor =[0,1,0,0.4], specularColor=[1, 1, 1])
        self.goal_visualization_id =p.createMultiBody(baseVisualShapeIndex=goal_visualization_shape_id)

        # initalize robot
        startPos = [0,0,0]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.iiwaId = p.loadURDF("kuka_iiwa/model.urdf",startPos, startOrientation, useFixedBase=True)# there are other configurations under this directory as well
        
        # get joint number 
        self.joint_num = p.getNumJoints(self.iiwaId)
        
        # getting lower and upper joint limit
        self.joint_limit =np.zeros((self.joint_num,2))
        for i in range(self.joint_num):
            joint_info = p.getJointInfo(self.iiwaId,i)
            self.joint_limit[i]= np.array(joint_info[8:10])

        self.reset()

        self.episode_length =2400
        self.threshold =1e-4

    def reset(self):
         # randomly setting a valid target
        target_joint_pos = np.random.uniform(self.joint_limit[:,0],self.joint_limit[:,1])

        self.setPositionStates(target_joint_pos)
        
        # get target end effector state in task space
        target_eef_states = p.getLinkState(self.iiwaId,6)

        self.target_eef_positions=np.array(target_eef_states[4])
        self.target_eef_orientations = np.array(target_eef_states[5])

        # set visualization to the location 
        p.resetBasePositionAndOrientation(self.goal_visualization_id, self.target_eef_positions, self.target_eef_orientations)

        #randomly initialize robot to a valid state
        initial_joint_pos = np.random.uniform(self.joint_limit[:,0],self.joint_limit[:,1])
        self.setPositionStates(initial_joint_pos)
        
        #reset time to 0
        self.steps=0
        self.done =False

        return np.concatenate((self.getStates(),self.target_eef_positions, self.target_eef_orientations))
    
    # get the states of iiwa joint positions and velocities
    def getStates(self):
        joint_states= p.getJointStates(self.iiwaId,jointIndices=[i for i in range(self.joint_num)])
        state_vec = np.array([joint_state[0:2] for joint_state in joint_states]).flatten()
        return state_vec
    
    # reset joint position state to given vectors, velocities are set to 0
    def setPositionStates(self,states):
        for num in range(self.joint_num):
            p.resetJointState(self.iiwaId, num, states[num])
    
    def getEEFPose(self):
        pass
        
    def setEEFPose(self):
        pass
    
    def getTargetEEFPose(self):
        pass

    def setTargetEEFPose(self):
        pass

    def resetTime(self):
        self.steps=0

    def step(self,action):
        p.setJointMotorControlArray(self.iiwaId,jointIndices=[i for i in range(self.joint_num)], controlMode=p.POSITION_CONTROL, targetPositions = action)
        p.stepSimulation()

        state_vec = self.getStates()

        current_eef_state = p.getLinkState(self.iiwaId,6)

        # penalize error
        reward =-( np.linalg.norm(np.array(self.target_eef_positions)-np.array(current_eef_state[4]))+\
        np.linalg.norm(np.array(self.target_eef_orientations)-np.array(current_eef_state[5])) )

        if self.steps>=self.episode_length or abs(reward)<abs(self.threshold):
            self.done =True
            
        self.steps+=1

        return np.concatenate((state_vec,self.target_eef_positions, self.target_eef_orientations)), reward, self.done

    def close(self):
        p.disconnect()







