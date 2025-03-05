from dm_control import mjcf

# 定义可复用的机械臂模块
class Arm:
    def __init__(self, name):
        self.model = mjcf.RootElement(model=name)
        
        # 添加肩部
        self.shoulder = self.model.worldbody.add('body', name='shoulder')
        self.shoulder.add('geom', type='capsule', size=[0.05, 0.2], pos=[0, 0, 0.2], density="1")
        self.shoulder_joint = self.shoulder.add('joint', name='swing', type='ball', damping="5")
        
        # 添加前臂
        self.forearm = self.shoulder.add('body', name='forearm', pos=[0, 0, 0.4])
        self.forearm.add('geom', type='capsule', size=[0.04, 0.15], pos=[0, 0, 0.15], density="1")
        self.elbow_joint = self.forearm.add('joint', name='elbow', type='hinge', axis=[0, 1, 0], damping="5")
        
        # 添加执行器来控制关节
        self.model.actuator.add('position', name=f'{name}_shoulder_x', joint=self.shoulder_joint, gear=[1, 0, 0], kp="10")
        self.model.actuator.add('position', name=f'{name}_shoulder_y', joint=self.shoulder_joint, gear=[0, 1, 0], kp="10")
        self.model.actuator.add('position', name=f'{name}_shoulder_z', joint=self.shoulder_joint, gear=[0, 0, 1], kp="10")
        self.model.actuator.add('position', name=f'{name}_elbow', joint=self.elbow_joint, gear=[1], kp="10")

# 起点
model = mjcf.RootElement()
# 关闭重力
model.option.gravity = [0, 0, -9.81]

# 添加地面和光源
model.asset.add('texture', name='groundplane', type="2d", builtin='checker', mark="edge", width=300, height=300,
                rgb1=[0.2, 0.3, 0.4], rgb2=[0.5, 0.6, 0.7], markrgb=[0.8, 0.8, 0.8] )
model.asset.add('material', name='groundplane', texture='groundplane', texuniform="true", texrepeat="5 5", reflectance="0.5")
model.worldbody.add('geom', name='ground', type='plane', size=[5, 5, 1], material="groundplane")
model.worldbody.add('light', pos=[0, 0, 10], dir=[0, 0, -1])

# 添加左臂连接点
left_site = model.worldbody.add('site', pos=[-0.5, 0, 0.05])
# 添加右臂滑动底座
slider_base = model.worldbody.add('body', name='slider_base', pos=[0.5, 0, 0.05])
slider_base.add('geom', type='box', size=[0.1, 0.1, 0.02], rgba=[0.5, 0.5, 0.5, 0], contype=0, conaffinity=0, density=1e-7)
slider_base.add('joint', name='slider', type='slide', axis=[1, 0, 0], range=[-1, 1], damping="5")
right_mount = slider_base.add('site', pos=[0, 0, 0.02])
# 添加滑动底座的执行器
model.actuator.add("position", name="base_slider", joint="slider", gear=[1], kp="100")

# 实例化两个 Arm 并附加到场景
left_arm = Arm('left_arm')
right_arm = Arm('right_arm')
left_site.attach(left_arm.model)
right_mount.attach(right_arm.model)

# # 生成物理模型（不知道咋用）
# physics = mjcf.Physics.from_mjcf_model(model)

# # 找到名为 'box_body' 的 body
# box_body = model.worldbody.body["box_body"]

# # 修改几何体位置
# box_body.geom.pos = [0, 0, 0.5]

# # 删除自由关节
# del box_body.freejoint

# 保存模型到文件
with open("my_model.xml", "w") as f:
    f.write(model.to_xml_string())
