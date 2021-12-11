import mujoco_py
import os
'''
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')

model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

viewer = mujoco_py.MjViewer(sim)

'''
xml_path = os.path.join(os. getcwd(), 'baxter','robot.xml') 
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)

viewer = mujoco_py.MjViewer(sim)

print(sim.data.qpos)


#sim.set_state_from_flattened([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])

t=0
while t<100000:
    
    #sim.forward()
    if t>10000:
        sim.data.ctrl[2]=100
    sim.step()
    viewer.render()
    t+=1


print(sim.data.qpos)