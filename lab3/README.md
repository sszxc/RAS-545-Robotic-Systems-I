# Digital Twin for Magician Robot

## Simulate in MuJoCo
1. Use Fusion 360 to open stp file, decompose link (redundant parts removed), export stl;
2. Use MeshLab to import stl, simplify the number of faces, export _binary_ STL;
3. Use PyMJCF to modify the relative position, rotation, and joint configuration of the robot in the xml file.

It is important to note that the Magician robotic arm’s joint 3 actually has two rotational axes. However, for simplification, it has been represented as a single joint here, with appropriately adjusted D-H parameters to better match the visual representation. As a result, the kinematics may not fully correspond to the actual robot.

![magician-links](./media/magician-links.png)

![stp-components](./media/fusion.png)

![magician-mujoco](./media/magician-mujoco.jpg)

## Inverse Kinematics

### Obtain Analytical Solution using SymPy

> lab3_dobot_ik.py `dobot_ik_analytical()`

Using the D-H parameters obtained from lab1 and the base-to-end transformation matrix, the symbolic expression for the forward kinematics of the robotic arm can be derived. By treating joint angles as variables and setting constraints between them (including joint angle limits and coupling relationships), the symbolic solution for the inverse kinematics of the robotic arm can be obtained.

### Obtain Numerical Solution using Scipy

> lab3_dobot_ik.py `dobot_ik_numerical()`

By taking the difference between the target position and the transformation matrix expression as the minimization objective, and setting up the optimization constraints (joint angle limits and coupling relationships), the inverse kinematics numerical solution of the robotic arm can also be obtained using `scipy.optimize.minimize` to solve the optimization problem.
