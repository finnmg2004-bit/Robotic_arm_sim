import os
import mujoco

path = os.path.expanduser("~/Projects/mujoco_menagerie/franka_fr3/scene.xml")
model = mujoco.MjModel.from_xml_path(path)

print("timestep:", model.opt.timestep)
for i in range(model.njnt):
    print("joint   ", i, model.joint(i).name, model.jnt_range[i])
for i in range(model.nu):
    print("actuator", i, model.actuator(i).name)
for i in range(model.nsite):
    print("site    ", i, model.site(i).name)
for i in range(model.nkey):
    print("keyframe", i, model.key(i).name)
