import pybullet as p
import time
import pybullet_data
import numpy as np
from pybullet_utils import bullet_client as bc

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
    def __init__(self, target_type, render=False):
        if target_type==None:
            raise Exception('Invalid target_type option')

        #connect to pybullet
        if render:
            self.physicsClient = bc.BulletClient(connection_mode=p.GUI)#P.GUI or p.DIRECT for non-graphical version
        else:
            self.physicsClient = bc.BulletClient(connection_mode=p.DIRECT)#P.GUI or p.DIRECT for non-graphical version
        # what Hz is simulated and controlled
        self.Hz = 30.

        #what type of target is training for 
        self.target_type = target_type
        self.physicsClient.setTimeStep(1./self.Hz)
        self.physicsClient.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        self.physicsClient.setGravity(0,0,-9.81)

        # create goal visualization
        goal_visualization_shape_id = self.physicsClient.createVisualShape(p.GEOM_CAPSULE, radius =0.1, length=0.2, rgbaColor =[0,1,0,0.4], specularColor=[1, 1, 1])
        self.goal_visualization_id =self.physicsClient.createMultiBody(baseVisualShapeIndex=goal_visualization_shape_id)

        # initalize robot
        startPos = [0,0,0]
        startOrientation = self.physicsClient.getQuaternionFromEuler([0,0,0])
        self.iiwaId = self.physicsClient.loadURDF("kuka_iiwa/model.urdf",startPos, startOrientation, useFixedBase=True)# there are other configurations under this directory as well
        
        # get joint number 
        self.joint_num = self.physicsClient.getNumJoints(self.iiwaId)
        
        # getting lower and upper joint limit
        self.joint_limit =np.zeros((self.joint_num,2))
        for i in range(self.joint_num):
            joint_info = self.physicsClient.getJointInfo(self.iiwaId,i)
            self.joint_limit[i]= np.array(joint_info[8:10])

        self.reset()

        # 10s of simulation
        self.episode_length =self.Hz*10

        self.threshold =1e-1

    def reset(self):
        if self.target_type=='Random':
            # randomly setting a valid target
            target_joint_pos = np.random.uniform(self.joint_limit[:,0],self.joint_limit[:,1])

            self.setPositionStates(target_joint_pos)
            
            # get target end effector state in task space
            target_eef_states = self.physicsClient.getLinkState(self.iiwaId,6)

            self.target_eef_positions=np.array(target_eef_states[4])
            self.target_eef_orientations = np.array(target_eef_states[5])

            # set visualization to the location 
            self.physicsClient.resetBasePositionAndOrientation(self.goal_visualization_id, self.target_eef_positions, self.target_eef_orientations)

            #randomly initialize robot to a valid state
            initial_joint_pos = np.random.uniform(self.joint_limit[:,0],self.joint_limit[:,1])
            
        elif self.target_type=='Point':
            # find a fixed target position
            # constant are from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
            self.target_eef_positions = np.array([0.2, 0.3, 0.7])
            self.target_eef_orientations = np.array(self.physicsClient.getQuaternionFromEuler([0,0,0])).flatten()
            
            #randomly initialize around 0 for all joints
            initial_joint_pos = np.random.uniform(self.joint_limit[:,0]/30,self.joint_limit[:,1]/30)
        
        self.setPositionStates(initial_joint_pos)

        # set visualization to the location 
        self.physicsClient.resetBasePositionAndOrientation(self.goal_visualization_id, self.target_eef_positions, self.target_eef_orientations)
        
        #reset time to 0
        self.steps=0
        self.done =False

        return np.concatenate((self.getStates(),self.target_eef_positions, self.target_eef_orientations))
    
    # get the states of iiwa joint positions and velocities
    def getStates(self):
        joint_states= self.physicsClient.getJointStates(self.iiwaId,jointIndices=[i for i in range(self.joint_num)])
        state_vec = np.array([joint_state[0:2] for joint_state in joint_states]).flatten()
        return state_vec
    
    # reset joint position state to given vectors, velocities are set to 0
    def setPositionStates(self,states):
        for num in range(self.joint_num):
            self.physicsClient.resetJointState(self.iiwaId, num, states[num])
    
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
        self.physicsClient.setJointMotorControlArray(self.iiwaId,jointIndices=[i for i in range(self.joint_num)], controlMode=p.POSITION_CONTROL, targetPositions = action)
        self.physicsClient.stepSimulation()

        state_vec = self.getStates()

        current_eef_state = self.physicsClient.getLinkState(self.iiwaId,6)

        # penalize error in orientation and position
        # for orientaion, if the difference between two quaternion is [0,0,0,1], means those two are aligned frame
        reward =-( np.linalg.norm(np.array(self.target_eef_positions)-np.array(current_eef_state[4]))+\
        np.linalg.norm(np.array(self.physicsClient.getDifferenceQuaternion(self.target_eef_orientations,current_eef_state[5]))-np.array([0,0,0,1])) )
        
        self.steps+=1
        '''
        # test termination condition
        if self.steps ==2400:
            print("Yes")

        if abs(reward)<abs(self.threshold):
            print("No")
        '''
        if self.steps>=self.episode_length or abs(reward)<abs(self.threshold):
            self.done =True
            # reach target 500 
            if abs(reward)<abs(self.threshold):
                reward =500

        return np.concatenate((state_vec,self.target_eef_positions, self.target_eef_orientations)), reward, self.done

    def close(self):
        self.physicsClient.disconnect()







