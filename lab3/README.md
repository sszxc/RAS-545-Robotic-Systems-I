# Digital Twin for Magician Robot

1. Use Fusion 360 to open stp file, decompose link (redundant parts removed), export stl;
2. Use MeshLab to import stl, simplify the number of faces, export _binary_ STL;
3. Use PyMJCF to modify the relative position, rotation, and joint configuration of the robot in the xml file.

![magician-links](./media/magician-links.png)

![magician-mujoco](./media/magician-mujoco.jpg)
