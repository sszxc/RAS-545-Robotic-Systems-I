# Forward Kinematics of myCobot Pro 600

![intro](./myCobot_Pro_600.png)

![result](./cobot_motion.gif)

- Requirements:
    ```
    pip install roboticstoolbox-python
    ```

        L1 = RevoluteDH(offset=0,        d=219.34,  a=0,      alpha=np.pi/2)
        L2 = RevoluteDH(offset=-np.pi/2, d=0,       a=-250,   alpha=0)
        L3 = RevoluteDH(offset=0,        d=0,       a=-250,   alpha=0)
        L4 = RevoluteDH(offset=np.pi/2,  d=-108,    a=0,      alpha=-np.pi/2)
        L5 = RevoluteDH(offset=0,        d=109.10,  a=0,      alpha=np.pi/2)
        L6 = RevoluteDH(offset=0,        d=-75.86,  a=0,      alpha=np.pi)
        
- D-H parameters:
    | Joint | θ | d | a | α |
    |-------|---|---|---|----|
    | 1-2 | θ1 | 219.34 | 0 | π/2 |
    | 2-3 | θ2 - π/2 | 0 | -250 | 0 |
    | 3-4 | θ3 | 0 | -250 | 0 |
    | 4-5 | θ4 + π/2 | -108 | 0 | -π/2 |
    | 5-6 | θ5 | 109.10 | 0 | π/2 |
    | 6-END | θ6 | -75.86 | 0 | π |

