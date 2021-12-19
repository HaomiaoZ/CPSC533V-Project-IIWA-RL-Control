import gym
from gym import spaces

import pybullet as p
import time
import pybullet_data
import numpy as np
from pybullet_utils import bullet_client as bc
from torch.nn.functional import normalize

class IIWAEnvGym(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human','rgb_array']}
  Hz = 30.
  episode_length =10*Hz
  pose_error_threshold = 1e-1


  def __init__(self, target_type, render=False):
    super(IIWAEnvGym, self).__init__()

    if target_type==None:
            raise Exception('Invalid target_type option')

    #connect to pybullet
    if render:
        self.physicsClient = bc.BulletClient(connection_mode=p.GUI)#P.GUI or p.DIRECT for non-graphical version
    else:
        self.physicsClient = bc.BulletClient(connection_mode=p.DIRECT)#P.GUI or p.DIRECT for non-graphical version
    
    #what type of target is training for 
    self.target_type = target_type
    self.physicsClient.setTimeStep(1./self.Hz)
    self.physicsClient.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    self.physicsClient.setGravity(0,0,-9.81)

    # create goal visualization
    goal_visualization_shape_id = self.physicsClient.createVisualShape(p.GEOM_CAPSULE, radius =0.1, length=0.2, rgbaColor =[0,1,0,0.4], specularColor=[1, 1, 1])
    self.goal_visualization_id =self.physicsClient.createMultiBody(baseVisualShapeIndex=goal_visualization_shape_id)

    # since no need to move the range viz, don't add to field
    if self.target_type=="Box" or self.target_type=="Box_pos" :
        self.box_half_extents = np.array([0.075,0.2,0.2])
        self.box_center = np.array([0.6,0,0.4])
        goal_range_visualization_shape_id = self.physicsClient.createVisualShape(p.GEOM_BOX, halfExtents = self.box_half_extents, rgbaColor =[1,0,0,0.2], specularColor=[1, 1, 1])
        goal_range_visualization_id =self.physicsClient.createMultiBody(baseVisualShapeIndex=goal_range_visualization_shape_id)
        self.physicsClient.resetBasePositionAndOrientation(goal_range_visualization_id, self.box_center, self.physicsClient.getQuaternionFromEuler([0,0,0]))
    # since no need to move the range viz, don't add to field
    if self.target_type=="Random" :
        goal_range_visualization_shape_id = self.physicsClient.createVisualShape(p.GEOM_SPHERE, radius= 0.82, rgbaColor =[1,0,0,0.2], specularColor=[1, 1, 1])
        goal_range_visualization_id =self.physicsClient.createMultiBody(baseVisualShapeIndex=goal_range_visualization_shape_id)
        self.physicsClient.resetBasePositionAndOrientation(goal_range_visualization_id, [0, 0, 0.36], self.physicsClient.getQuaternionFromEuler([0,0,0]))
    # initalize robot
    startPos = [0,0,0]
    startOrientation = self.physicsClient.getQuaternionFromEuler([0,0,0])
    self.iiwaId = self.physicsClient.loadURDF("kuka_iiwa/model.urdf",startPos, startOrientation, useFixedBase=True)# there are other configurations under this directory as well
    
    # get joint number 
    self.joint_num = self.physicsClient.getNumJoints(self.iiwaId)
    
    # getting lower and upper joint limit, joint velocity limit 
    self.joint_position_limit =np.zeros((self.joint_num,2))
    self.joint_velocity_limit = np.zeros(self.joint_num)
    for i in range(self.joint_num):
        joint_info = self.physicsClient.getJointInfo(self.iiwaId,i)
        self.joint_position_limit[i]= np.array(joint_info[8:10])
        self.joint_velocity_limit[i] = joint_info[11]
    
    # Define action and observation space
    # They must be gym.spaces objects
    # IIWA 14 R820 model
    # TODO: figure out how to limit velocity in simulation, currently contorl mode will not consider that in pybullet (double check)
    # normalize the action space and observation space
    self.action_space = spaces.Box(low=-1, high=1,shape = (self.joint_num,), dtype = np.float32)
    self.observation_space = spaces.Box(low=-1, high=1, shape =(len(self.joint_position_limit[:,0])*2+7,),dtype =np.float32)
    '''
    self.observation_space = spaces.Box(low=np.concatenate((self.joint_position_limit[:,0],-self.joint_velocity_limit*10,np.array([-1,-1,-1.5]),-np.ones(4))),\
         high=np.concatenate((self.joint_position_limit[:,1],self.joint_velocity_limit*10,np.array([1,1,1.5]),np.ones(4))),\
              dtype=np.float64)
    '''

  def step(self, action):
    action =np.multiply(action, self.joint_position_limit[:,1])
    self.physicsClient.setJointMotorControlArray(self.iiwaId,jointIndices=[i for i in range(self.joint_num)], controlMode=p.POSITION_CONTROL, targetPositions = action)
    self.physicsClient.stepSimulation()

    state_vec = self.getStates()

    current_eef_state = self.physicsClient.getLinkState(self.iiwaId,6)

    # penalize error in orientation and position
    # for orientaion, if the difference between two quaternion is [0,0,0,1], means those two are aligned frame
    reward =-( np.linalg.norm(np.array(self.target_eef_positions)-np.array(current_eef_state[4]))+\
    np.linalg.norm(np.array(self.physicsClient.getDifferenceQuaternion(self.target_eef_orientations,current_eef_state[5]))-np.array([0,0,0,1])) )
    
    if self.target_type=="Box_pos":
        reward = -np.linalg.norm(np.array(self.target_eef_positions)-np.array(current_eef_state[4]))
    
    self.steps+=1
    '''
    # test termination condition
    if self.steps ==2400:
        print("Yes")

    if abs(reward)<abs(self.pose_error_threshold):
        print("No")
    '''
    if self.steps>=self.episode_length or abs(reward)<abs(self.pose_error_threshold):
        self.done =True
        # reach target 500 
        if abs(reward)<abs(self.pose_error_threshold):
            reward =500
    observation = self.__normalize_observation__(np.concatenate((state_vec,self.target_eef_positions, self.target_eef_orientations)))
    return observation, reward, self.done,{}

  def reset(self):
    if self.target_type=="Random":
        # randomly setting a valid target
        target_joint_pos = np.random.uniform(self.joint_position_limit[:,0],self.joint_position_limit[:,1])

        self.setPositionStates(target_joint_pos)
        
        # get target end effector state in task space
        target_eef_states = self.physicsClient.getLinkState(self.iiwaId,6)

        self.target_eef_positions=np.array(target_eef_states[4])
        self.target_eef_orientations = np.array(target_eef_states[5])

        #randomly initialize robot to a valid state
        initial_joint_pos = np.random.uniform(self.joint_position_limit[:,0],self.joint_position_limit[:,1])
        
    elif self.target_type=="Point":
        # find a fixed target position
        # constant are from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
        self.target_eef_positions = np.array([0.2, 0.3, 0.7])
        self.target_eef_orientations = np.array(self.physicsClient.getQuaternionFromEuler([0,0,0])).flatten()
        
        #randomly initialize around 0 for all joints
        initial_joint_pos = np.random.uniform(np.deg2rad(-np.ones(self.joint_num)*5),np.deg2rad(np.ones(self.joint_num)*5))
    
    elif self.target_type=="Box" or self.target_type=="Box_pos":
        valid_position=False
        while not valid_position:
            # randomize orientation by randomize euler angle
            target_eef_pos = np.random.uniform(self.box_center-self.box_half_extents,self.box_center+self.box_half_extents)
            target_eef_ori = np.array(self.physicsClient.getQuaternionFromEuler(np.random.uniform(np.zeros(3),np.ones(3)*2*np.pi))).flatten()
            valid_position = self.__is_valid__(target_eef_pos,target_eef_ori)

        self.target_eef_positions = target_eef_pos
        self.target_eef_orientations = target_eef_ori

        #randomly initialize around 0 for all joints
        initial_joint_pos = np.random.uniform(np.deg2rad(-np.ones(self.joint_num)*5),np.deg2rad(np.ones(self.joint_num)*5))

    else:
        raise Exception("Target type is not defined or incorrect")

    self.setPositionStates(initial_joint_pos)

    # set visualization to the location 
    self.physicsClient.resetBasePositionAndOrientation(self.goal_visualization_id, self.target_eef_positions, self.target_eef_orientations)
    
    #reset time to 0
    self.steps=0
    self.done =False

    observation = self.__normalize_observation__(np.concatenate((self.getStates(),self.target_eef_positions, self.target_eef_orientations)))
    return observation  # reward, done, info can't be included

  # thanks to https://www.etedal.net/2020/04/pybullet-panda_2.html 
  def render(self, mode='human'):
    if mode =="human":
        return np.array([])
    elif mode=="rgb_array":
        width =1920
        height =1080
        view_matrix = self.physicsClient.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0,0,0.4],
                                                                distance=2,
                                                                yaw=125,
                                                                pitch=-15,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self.physicsClient.computeProjectionMatrixFOV(fov=60,
                                                    aspect=float(width) /height,
                                                    nearVal=0.1,
                                                    farVal=100.0)
        (_, _, px, _, _) = self.physicsClient.getCameraImage(
		width=width, height=height, viewMatrix=view_matrix,
			projectionMatrix=proj_matrix,
			renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    
  def close (self):
    self.physicsClient.disconnect()
  
  def resetTime(self):
    self.steps=0
# get the states of iiwa joint positions and velocities
  def getStates(self):
    joint_states= self.physicsClient.getJointStates(self.iiwaId,jointIndices=[i for i in range(self.joint_num)])
    state_vec = np.concatenate((np.array([joint_state[0] for joint_state in joint_states]).flatten(),\
        np.array([joint_state[1] for joint_state in joint_states]).flatten()))
    return state_vec
    
  # reset joint position state to given vectors, velocities are set to 0
  def setPositionStates(self,states):
    for num in range(self.joint_num):
        self.physicsClient.resetJointState(self.iiwaId, num, states[num])
    
  # is valid can only be used during reset since it will set velocity to 0!
  # calculate target joint pos, then check whether the yielded task space pose is within error rate, since IK might provide solution that satisfy pose but a bit far away
  def __is_valid__(self, target_eef_pos,target_eef_ori):

    target_joint_pos = self.physicsClient.calculateInverseKinematics(self.iiwaId,6,target_eef_pos,target_eef_ori,\
    [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05],\
        [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05],\
            [5.8, 4, 5.8, 4, 5.8, 4, 6],\
            [0, 0, 0, 0.5 * np.pi, 0, -np.pi * 0.5 * 0.66, 0])
    
    cur_joint_states = self.getStates()
    # first 7 elements are current position
    cur_joint_pos = cur_joint_states[0:8]

    # set to target position and calculate error
    self.setPositionStates(target_joint_pos)

    current_eef_state = self.physicsClient.getLinkState(self.iiwaId,6)

    error_in_pose =np.linalg.norm(np.array(target_eef_pos)-np.array(current_eef_state[4]))+\
    np.linalg.norm(np.array(self.physicsClient.getDifferenceQuaternion(target_eef_ori,current_eef_state[5]))-np.array([0,0,0,1]))

    #reser to positions before calling this function
    self.setPositionStates(cur_joint_pos)

    # check whether it is reachable
    return error_in_pose<self.pose_error_threshold

  # normalize state+position and orientation of goal to observation lie between [-1 to 1] need to normalize joint position, velocity and z position
  def __normalize_observation__(self,observation):
    observation[0:self.joint_num] = np.divide(observation[0:self.joint_num],self.joint_position_limit[:,1])
    observation[self.joint_num:self.joint_num*2] = np.divide(observation[self.joint_num:self.joint_num*2],self.joint_velocity_limit*10)
    # assume z is in [-0.3, 1.5], normalize to [-1,1]
    observation[16] = (observation[16]-0.6)/0.9

    return observation

  #helper function for PD controller to calculate action
  def getPhysicsClient(self):
      return self.physicsClient
  def getIIWAId(self):
      return self.iiwaId
  def getJointPositionLimit(self):
      return self.joint_position_limit[:,1]