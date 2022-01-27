
# %% Define some model props
import numpy as np
import sys
# Reduced MBD model properties
n_modes = 6
static_modes_end = np.arange(6) # [1]

# FEM model properties and geometry
res = []

# %%
from flexmbd.fem import BeamFem

from flexmbd.helpers import Material, Section
from flexmbd.utils import rot

def generate(n_elem, l, h, E, nu, rho, Fy):
    mat = Material(rho, E, nu)
    sec = Section([h])
    fem = BeamFem.straight(sec, mat, l, n_elem)

    # %% Try to solve a simple static problem

    fem.set_boundary_conditions([0], [[0, 1, 2, 3, 4, 5]])
    K = fem.stiffness().toarray()
    F = np.zeros((K.shape[0],))
    F[-5] = Fy

    u = np.linalg.solve(K, F)
    #  Basic results
    print(
        f"FEM: end displacement is {u[-5] * 1000:.5g} mm, rotation {u[-1] * 180 / np.pi:.4g} deg"
    )
    # %% Now perform CB reduction and solve system
    # Recreate without boundary conditions
    from flexmbd.fbodies import (
        ReducedModelCraigBampton,
        ReducedModelCraigBamptonOrth,
        MassIntegralsMatrix,
        SidByMassIntegrals,
        FBodySidIntegrals,
    )

    fem = BeamFem.straight(sec, mat, l, n_elem)

    rm = ReducedModelCraigBamptonOrth(
        fem,
        cb_nodes=[0, n_elem],
        cb_dofs=[np.arange(6), static_modes_end],
        n_modal=n_modes,
    )

    mi = MassIntegralsMatrix(rm)
    sid = SidByMassIntegrals(
        rm, mi, out_nodes=[0, n_elem], rayleigh_damping=[3 * 100, 100 * 1e-4]
    )
    x0 = l / 2
    fb = FBodySidIntegrals(sid, q0=np.array([x0, 0, 0, 1, 0, 0, 0]))

    # %% Buiid system
    from flexmbd.bodies import RBody
    from flexmbd.forces import ForcePoint
    from flexmbd.joints import JointPoint, JointPerpend1, JointSimple

    ground = RBody(1, np.eye(3))
    joints_loc = np.zeros((3,))
    joints = [
        JointSimple(ground),  # Fix ground
        JointPoint(ground, fb, joints_loc),  # Fixed joint between dummy and flex
        # JointPerpend1(ground, fb, [1, 0, 0], [0, 1, 0], joints_loc),
        # JointPerpend1(ground, fb, [1, 0, 0], [0, 0, 1], joints_loc),
        # JointPerpend1(ground, fb, [0, 1, 0], [0, 0, 1], joints_loc),
        JointPerpend1(fb, ground, [1, 0, 0], [0, 1, 0], joints_loc),
        JointPerpend1(fb, ground, [1, 0, 0], [0, 0, 1], joints_loc),
        JointPerpend1(fb, ground, [0, 1, 0], [0, 0, 1], joints_loc),
    ]

    # Apply force at body end
    force_loc = np.array([l, 0.0, 0.0])


    def smooth_step(t, v_max, t_max_start, t_max_end, t_zero):
        if t < t_max_start:
            return 0.5 * v_max * (1 - np.cos(np.pi * t / t_max_start))
        elif t < t_max_end:
            return v_max
        elif t < t_zero:
            return (
                0.5 * v_max * (1 + np.cos(np.pi * (t - t_max_end) / (t_zero - t_max_end)))
            )
        return 0


    frc = ForcePoint(
        fb, lambda t: np.array([0, smooth_step(t, Fy, 0.5, 100.0, 100.0), 0]), force_loc
    )

    from flexmbd.mbs import Mbs

    sys = Mbs([ground], [fb], joints=joints, forces=[frc],)
    y0 = sys.initial_position()

    # %% Integrate
    from scipy.integrate import solve_ivp

    res = solve_ivp(
        sys,
        (0, 1.0),
        y0,
        t_eval=np.linspace(0, 1.0, 1001),
        method="Radau",
        max_step=0.01 * 5,
        # rtol=1e-6,
        # atol=1e-9,
    )
    # %%
    rotations = []
    import matplotlib.pyplot as plt

    ny = res.y.shape[0] // 2 + 1
    r_end = np.empty((3, res.y.shape[1]))
    for i in range(res.y.shape[1]):
        u_last = sid.nodes[-1].origin @ res.y[14:ny, i]
        A_fb = rot(res.y[10:14, i])
        r_fb = res.y[7:10, i]
        r = r_fb + A_fb @ u_last
        r_end[:, i] = r
        #rotations.append()
    ## r_end aka all displacements. [1] is the y-axis displacement

    plt.plot(res.t, r_end[1])
    #plt.show()

    # Displacement
    AP_last = sid.nodes[-1].AP @ res.y[14:ny, -1]

    A = A_fb @ AP_last
    print(
        f"MBD: end displacement is {r[1] * 1000:.5g} mm, rotation {np.arcsin(A[1, 0]) * 180 / np.pi:.4g} deg"
    )


    print(len(res.y[1]))
    return [res.y, r_end[1]]


#generate(40,1,0.02,2e11,0.3,7801,500.0)
# %%
