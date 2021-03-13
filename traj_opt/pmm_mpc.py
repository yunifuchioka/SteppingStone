import numpy as np
from numpy import pi
import casadi as ca

from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

plt.style.use('seaborn')

TOL = 1E-12  # value under which a number is considered to be zero

nr = 3  # dimension of centroid configuration vector
nj = 4  # number of contact points
nj3 = 3 * nj  # dimension of vectors associated with contact points

leg_len = np.array([0.45, 0.5])  # thigh and calf lengh respectively
torso_size = np.array([0, 0.2, 0.3])  # x,y,z length of torso
foot_size = np.array([0.2, 0, 0])  # x,y,z length of feet

m = 39.3  # mass of torso
zc = leg_len[0] + leg_len[1] + torso_size[2] - 0.1  # nominal height of torso
g = np.array([0, 0, -9.81])  # gravitational acceleration
mu = 0.8  # friciton coefficient

Rs_r = np.eye(3)
# Rs_r = Rotation.from_rotvec(37*pi/180 * np.array([0, 1, 0])).as_matrix()
ps_r = np.zeros((3, 1))

Rs_l = Rs_r
# Rs_l = Rotation.from_rotvec(10*pi/180 * np.array([0, 1, 0])).as_matrix()
ps_l = ps_r
# ps_l = np.array([0, 0, 0.3])[:,None]

tf = 20  # final time of interval
N = int(tf * 4)  # number of hermite-simpson finite elements. Total # of points = 2*N+1
t = np.linspace(0, tf, 2 * N + 1)  # discretized time

# objective cost weights
Qr = np.array([1, 1, 1000])
Qrd = np.array([1, 1, 1])
Qc = np.full((nj3), 1000)
Qcd = np.full((nj3), 10)
RF = np.full((nj3), 0.0001)

# footstep planner parameters
stride = np.array([0.4, torso_size[1], 0.3])  # forward step length, leg width, max foot height
stride_std = np.array([0.0, 0.0, 0.0])  # random std dev of step x and y locations
T_step = 0.5  # step period
duty = 0.5  # duty cycle - proportion of gait where foot is on the ground

# desired trajectory planning #############################################################
num_steps = int(tf / T_step)  # number of footsteps to take
# matrix that specifies footsteps timing and locations
footstep_plan = np.array([
    np.linspace(0, tf, num_steps),  # time
    np.hstack((np.zeros(2),  # x positions
               np.linspace(0.0, (num_steps - 4) * stride[0], num_steps - 4),
               np.full(2, (num_steps - 4) * stride[0]),
               )),
    np.array([-stride[1] / 2 + stride[1] * (i % 2) for i in range(num_steps)])  # y positions
])

# add random noise
footstep_plan[1, 2:-1] += np.random.normal(loc=0, scale=stride_std[0], size=max(footstep_plan[1].shape) - 3)
footstep_plan[2, 2:-1] += np.random.normal(loc=0, scale=stride_std[1], size=max(footstep_plan[2].shape) - 3)

# interpolations points for feet: rows are time, x, y, z
right_traj_points = footstep_plan[:, footstep_plan[2, :] <= 0]
left_traj_points = footstep_plan[:, footstep_plan[2, :] >= 0]
right_traj_points = np.concatenate((
    right_traj_points,
    (ps_r[2] - Rs_r[0, 2] / Rs_r[2, 2] * (right_traj_points[1, :] - ps_r[0])
     - Rs_r[1, 2] / Rs_r[2, 2] * (right_traj_points[2, :] - ps_r[1]))[None, :]
), axis=0)
left_traj_points = np.concatenate((
    left_traj_points,
    (ps_l[2] - Rs_l[0, 2] / Rs_l[2, 2] * (left_traj_points[1, :] - ps_l[0])
     - Rs_l[1, 2] / Rs_l[2, 2] * (left_traj_points[2, :] - ps_l[1]))[None, :]
), axis=0)

# add endpoints if they are missing
if right_traj_points[0, 0] != 0:
    right_traj_points = np.concatenate((
        np.insert(right_traj_points[1:, 0], 0, 0)[:, None],
        right_traj_points
    ), axis=1)
if left_traj_points[0, 0] != 0:
    left_traj_points = np.concatenate((
        np.insert(left_traj_points[1:, 0], 0, 0)[:, None],
        left_traj_points
    ), axis=1)
if right_traj_points[0, -1] != tf:
    right_traj_points = np.concatenate((
        right_traj_points,
        np.insert(right_traj_points[1:, -1], 0, tf)[:, None],
    ), axis=1)
if left_traj_points[0, -1] != tf:
    left_traj_points = np.concatenate((
        left_traj_points,
        np.insert(left_traj_points[1:, -1], 0, tf)[:, None],
    ), axis=1)

# specify when foot liftoff should occur
right_liftoff_time = right_traj_points[0, :-1] * (1 - duty) + right_traj_points[0, 1:] * duty
left_liftoff_time = left_traj_points[0, :-1] * (1 - duty) + left_traj_points[0, 1:] * duty

# specify interpolation points corresponding to foot liftoff
right_liftoff_points = np.concatenate((
    right_liftoff_time[None, :],
    right_traj_points[1:, :-1]
), axis=0)
left_liftoff_points = np.concatenate((
    left_liftoff_time[None, :],
    left_traj_points[1:, :-1]
), axis=0)

# specify interpolation points corresponding to swing foot midpoint to specify foot height
right_swing_points = np.concatenate((
    (right_liftoff_time[1:-1] + right_traj_points[0, 2:-1])[None, :] / 2,
    (right_traj_points[1:3, 2:-1] + right_traj_points[1:3, 1:-2]) / 2,
    (right_traj_points[3, 2:-1] + right_traj_points[3, 1:-2])[None, :] / 2 + stride[2]
), axis=0)
left_swing_points = np.concatenate((
    (left_liftoff_time[1:-1] + left_traj_points[0, 2:-1])[None, :] / 2,
    (left_traj_points[1:3, 2:-1] + left_traj_points[1:3, 1:-2]) / 2,
    (left_traj_points[3, 2:-1] + left_traj_points[3, 1:-2])[None, :] / 2 + stride[2]
), axis=0)
# construct foot trajectory interpolator function
right_traj_points = np.concatenate((
    right_traj_points, right_liftoff_points, right_swing_points), axis=1)
left_traj_points = np.concatenate((
    left_traj_points, left_liftoff_points, left_swing_points), axis=1)
right_traj_points = right_traj_points[:, right_traj_points[0].argsort()]
left_traj_points = left_traj_points[:, left_traj_points[0].argsort()]
right_interp = interp1d(right_traj_points[0], right_traj_points[1:])
left_interp = interp1d(left_traj_points[0], left_traj_points[1:])

# desired foot trajectory
foot_traj = np.concatenate((
    right_interp(t),
    left_interp(t),
), axis=0)

# desired torso pose trajectory
r_traj = np.array([
    np.mean(np.vstack((foot_traj[0], foot_traj[3])), axis=0),
    np.mean(np.vstack((foot_traj[1], foot_traj[4])), axis=0),
    np.min(np.vstack((foot_traj[2], foot_traj[5])), axis=0) + zc
])
# desired foot location
c_des = np.vstack((foot_traj[:3], foot_traj[:3], foot_traj[3:], foot_traj[3:])) \
        + np.hstack((Rs_r @ foot_size / 2, Rs_r @ -foot_size / 2,
                     Rs_l @ foot_size / 2, Rs_l @ -foot_size / 2))[:, None]

# torso pose and velocity trajectory
spline = CubicSpline(t, r_traj, axis=1)
xr_des = np.vstack((spline(t), spline(t, 1)))

# guess contact force trajectory
# initialize as Fx=Fy=0, Fz=(torso mass)/(# contact points)
F_des = np.tile(-m * g / 4, (2 * N + 1, nj)).T
# desired contact point distances from ground surface origin (in global frame)
c_des_rel = c_des - np.vstack((ps_r, ps_r, ps_l, ps_l))
# normal distance from ground surface for each time
c_normal_dist = np.vstack((Rs_r.T[2, :] @ c_des_rel[0:3, :], Rs_r.T[2, :] @ c_des_rel[3:6, :],
                           Rs_l.T[2, :] @ c_des_rel[6:9, :], Rs_l.T[2, :] @ c_des_rel[9:12, :]))
# set Fz=0 when c_des not in contact with ground
F_des[2::3, :][np.abs(c_normal_dist) >= TOL] = 0


# derive dynamic equation of motion
def derive_dynamics():
    xr = ca.SX.sym('xr', nr * 2)
    uF = ca.SX.sym('uF', nj3)

    F_net = ca.sum2(uF.reshape((3, nj)))
    rddot = F_net / m + g

    f_sym = ca.vertcat(xr[nr:], rddot)
    f = ca.Function('f', [xr, uF], [f_sym])

    return f


f = derive_dynamics()

# trajectory optimization #############################################################
opti = ca.Opti()

# decision variables
XR = opti.variable(nr * 2, 2 * N + 1)  # configuration + velocity of COM position
XC = opti.variable(nj3, 2 * N + 1)  # configuration of contact points
UF = opti.variable(nj3, 2 * N + 1)  # contact point force

# cost
# simpson quadrature coefficients, to be used to compute integrals
simp = np.empty((1, 2 * N + 1))
simp[0, ::2] = 2
simp[0, 1::2] = 4
simp[0, 0], simp[0, -1] = 1, 1

J = 0.0
for i in range(2 * N + 1):
    for r_idx in range(nr):
        J += Qr[r_idx] * simp[0][i] * (XR[r_idx, i] - xr_des[r_idx, i]) ** 2
        J += Qrd[r_idx] * simp[0][i] * (XR[nr + r_idx, i] - xr_des[nr + r_idx, i]) ** 2

    for c_idx in range(nj3):
        J += Qc[c_idx] * simp[0][i] * (XC[c_idx, i] - c_des[c_idx, i]) ** 2
        if i != 0:
            J += Qcd[c_idx] * simp[0][i] * (XC[c_idx, i] - XC[c_idx, i - 1]) ** 2

    for F_idx in range(nj3):
        J += RF[F_idx] * simp[0][i] * (UF[F_idx, i]) ** 2

opti.minimize(J)

# initial condition constraint
opti.subject_to(XR[:, 0] == xr_des[:, 0])
opti.subject_to(XC[:, 0] == c_des[:, 0])

for i in range(2 * N + 1):
    # point mass dynamics constraints imposed through Hermite-Simpson
    if i % 2 != 0:
        # for each finite element:
        xr_left, xr_mid, xr_right = XR[:, i - 1], XR[:, i], XR[:, i + 1]
        uF_left, uF_mid, uF_right = UF[:, i - 1], UF[:, i], UF[:, i + 1]
        f_left, f_mid, f_right = \
            f(xr_left, uF_left), f(xr_mid, uF_mid), f(xr_right, uF_right)

        # interpolation constraints
        opti.subject_to(
            # equation (6.11) in Kelly 2017
            xr_mid == (xr_left + xr_right) / 2.0 + tf / N * (f_left - f_right) / 8.0)

        # collocation constraints
        opti.subject_to(
            # equation (6.12) in Kelly 2017
            tf / N * (f_left + 4 * f_mid + f_right) / 6.0 == xr_right - xr_left)

    # all other constraints imposed at every timestep

    r_i = XR[:nr, i]  # COM position at current timestep
    F_i = UF[:, i].reshape((3, nj))  # contact forces at current timestep, 3xnj matrix
    c_i = XC[:nj3, i].reshape((3, nj))  # global contact pos at current timestep, 3xnj matrix
    c_i_prev = XC[:nj3, i - 1].reshape((3, nj))  # global contact pos at prev timestep, 3xnj matrix
    c_rel_i = c_i - r_i  # contact pos rel to COM at current timestep, 3xnj matrix

    # zero angular momentum constraint
    torque_i = ca.MX.zeros(3, nj)
    for j in range(nj):
        torque_i[:, j] = ca.cross(c_rel_i[:, j], F_i[:, j])
    opti.subject_to(ca.sum2(torque_i) == np.zeros((3, 1)))

    # foot size and orientation constraint
    # TODO: ignore terrain rotations in z axis
    opti.subject_to(c_i[:, 0] - c_i[:, 1] == Rs_r @ foot_size)
    opti.subject_to(c_i[:, 2] - c_i[:, 3] == Rs_l @ foot_size)

    # foot positions and reaction forces and surface coordinates
    idx_rl = int(nj / 2)  # index representing switch between right and left feet
    # ground reaction forces in ground surface frame
    Fs_i = ca.horzcat(Rs_r.T @ F_i[:, :idx_rl], Rs_l.T @ F_i[:, idx_rl:])
    # foot locations in ground surface frame
    cs_i = ca.horzcat(Rs_r.T @ (c_i[:, :idx_rl] - ps_r), Rs_l.T @ (c_i[:, idx_rl:] - ps_l))
    # foot velocities in ground surface frame
    cds_i = ca.horzcat(Rs_r.T @ (c_i[:, :idx_rl] - c_i_prev[:, :idx_rl]),
                       Rs_l.T @ (c_i[:, idx_rl:] - c_i_prev[:, idx_rl:]))

    # friciton cone constraints in surface coordinates
    opti.subject_to(Fs_i[2, :] >= np.zeros(Fs_i[2, :].shape))
    opti.subject_to(opti.bounded(-mu * Fs_i[2, :], Fs_i[0, :], mu * Fs_i[2, :]))
    opti.subject_to(opti.bounded(-mu * Fs_i[2, :], Fs_i[1, :], mu * Fs_i[2, :]))

    # contact constraints in surface coordinates
    opti.subject_to(cs_i[2, :] >= np.zeros(cs_i[2, :].shape))
    opti.subject_to(opti.bounded(-TOL, cs_i[2, :] * Fs_i[2, :], TOL))
    if i != 0:
        opti.subject_to(opti.bounded(-TOL, cds_i[0, :] * Fs_i[2, :], TOL))
        opti.subject_to(opti.bounded(-TOL, cds_i[1, :] * Fs_i[2, :], TOL))

    # joint constraint planar approximation
    '''
    opti.subject_to(c_rel_i[0,0]-foot_size[0]/2-c_rel_i[2,0] <= 1.5)
    opti.subject_to(c_rel_i[0,2]-foot_size[0]/2-c_rel_i[2,2] <= 1.5)
    opti.subject_to(-c_rel_i[0,0]+foot_size[0]/2-c_rel_i[2,0] <= 1.5)
    opti.subject_to(-c_rel_i[0,2]+foot_size[0]/2-c_rel_i[2,2] <= 1.5)
    '''

# initial guess for decision variables
XR_guess = xr_des
XC_guess = c_des
UF_guess = F_des
# XC_guess = np.repeat(c_des.reshape((1,nj3),order='F'),2*N+1,axis=0).T
opti.set_initial(XR, XR_guess)
opti.set_initial(XC, XC_guess)
opti.set_initial(UF, UF_guess)

# solve NLP
p_opts = {}
s_opts = {'print_level': 5}
opti.solver('ipopt', p_opts, s_opts)
sol = opti.solve()

# extract NLP output as numpy array
XR_sol = np.array(sol.value(XR))[:nr, :]
XC_sol = np.array(sol.value(XC))
UF_sol = np.array(sol.value(UF))

# trajectory to be used for RL
pmm_traj_raw = np.vstack((
    XC_sol[3, :],  # right heel x
    XC_sol[5, :],  # right heel z
    XC_sol[9, :],  # left heel x
    XC_sol[11, :],  # left heel z
    XR_sol[0, :],  # COM x
    XR_sol[2, :],  # COM z
))

t_RL = np.linspace(0, tf, tf * 60)  # timepoints with 60Hz frequency, to be consistent with RL
pmm_traj_interpolator = CubicSpline(t, pmm_traj_raw, axis=1)
pmm_traj = pmm_traj_interpolator(t_RL)  # pmm trajectory with time spacing that is consistent with RL

"""
# check what trajectory looks like
plt.plot(t_RL, pmm_traj[0::2].T)
plt.title("x trajectories")
plt.show()
plt.plot(t_RL, pmm_traj[1::2].T)
plt.title("y trajectories")
plt.show()
"""

np.save('../trajectories/pmm_traj3.npy', pmm_traj)

"""

# inverse kinematics ###################################################################

nq = 10  # dimension of joint configuration vector

# Denavit Hartenberg parameters for a leg
theta_offset = np.array([0.0, -pi / 2, 0.0, 0.0, 0.0])
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
a = np.array([0.0, 0.0, leg_len[0], leg_len[1], 0.0])
alpha = np.array([-pi / 2, -pi / 2, 0.0, 0.0, 0.0])

# "default" joint angle configuration
q_guess = np.array([
    0, 0, -0.05, 0.1, -0.05,
    0, 0, -0.05, 0.1, -0.05
])

# joint angle limits
q_bounds = np.array([
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([1E-6, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([-pi, pi]),
    np.array([1E-6, pi]),
    np.array([-pi, pi])
])

# cartesian basis vectors
i0 = np.array([1.0, 0.0, 0.0])[:, None]
j0 = np.array([0.0, 1.0, 0.0])[:, None]
k0 = np.array([0.0, 0.0, 1.0])[:, None]


# derive functions for rotation matrices and homogeneous transformation matrices
def derive_coord_trans():
    s = ca.SX.sym('s', 3)
    th = ca.SX.sym('th')
    th_offset = ca.SX.sym('th_offset')
    d = ca.SX.sym('d')
    a = ca.SX.sym('a')
    alpha = ca.SX.sym('alpha')

    # cross product matrix
    skew_sym = ca.SX(np.array([
        [0, -s[2], s[1]],
        [s[2], 0, -s[0]],
        [-s[1], s[0], 0]
    ]))

    # rotation matrix using Rodrigues' rotation formula
    R_sym = ca.SX.eye(3) + ca.sin(th) * skew_sym + (1 - ca.cos(th)) * skew_sym @ skew_sym
    R = ca.Function('R', [th, s], [R_sym])

    # homogeneous transformation matrix
    T_sym = ca.vertcat(
        ca.horzcat(R(th + th_offset, k0) @ R(alpha, i0), R(th + th_offset, k0) @ (a * i0) + d * k0),
        np.array([0.0, 0.0, 0.0, 1.0])[None, :]
    )
    T = ca.Function('T', [th, th_offset, d, a, alpha], [T_sym])

    return R, T


# derive forward kinematics for biped
def derive_forkin():
    q = ca.SX.sym('q', nq)

    T_base_right = T(pi / 2, 0, -torso_size[2], -torso_size[1] / 2, pi)
    T_base_left = T(pi / 2, 0, -torso_size[2], torso_size[1] / 2, pi)
    T5_c1 = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, -foot_size[0] / 2],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    T5_c2 = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, foot_size[0] / 2],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])

    T_r1 = T_base_right @ T(q[0], theta_offset[0], d[0], a[0], alpha[0])
    T_r2 = T_r1 @ T(q[1], theta_offset[1], d[1], a[1], alpha[1])
    T_r3 = T_r2 @ T(q[2], theta_offset[2], d[2], a[2], alpha[2])
    T_r4 = T_r3 @ T(q[3], theta_offset[3], d[3], a[3], alpha[3])
    T_r5 = T_r4 @ T(q[4], theta_offset[4], d[4], a[4], alpha[4])

    T_rc1 = T_r5 @ T5_c1
    T_rc2 = T_r5 @ T5_c2

    T_l1 = T_base_left @ T(q[5], theta_offset[0], d[0], a[0], alpha[0])
    T_l2 = T_l1 @ T(q[6], theta_offset[1], d[1], a[1], alpha[1])
    T_l3 = T_l2 @ T(q[7], theta_offset[2], d[2], a[2], alpha[2])
    T_l4 = T_l3 @ T(q[8], theta_offset[3], d[3], a[3], alpha[3])
    T_l5 = T_l4 @ T(q[9], theta_offset[4], d[4], a[4], alpha[4])

    T_lc1 = T_l5 @ T5_c1
    T_lc2 = T_l5 @ T5_c2

    p_feet_sym = ca.horzcat(T_rc1[0:3, 3], T_rc2[0:3, 3], T_lc1[0:3, 3], T_lc2[0:3, 3], )
    forkin_feet = ca.Function('forkin_feet', [q], [p_feet_sym])

    p_leg_sym = ca.horzcat(T_r1[0:3, 3], T_r3[0:3, 3], T_r4[0:3, 3],
                           T_l1[0:3, 3], T_l3[0:3, 3], T_l4[0:3, 3])
    forkin_leg = ca.Function('forkin_leg', [q], [p_leg_sym])

    return forkin_feet, forkin_leg


_, T = derive_coord_trans()
forkin_feet, forkin_leg = derive_forkin()

opti_kin = ca.Opti()
XQ = opti_kin.variable(nq, 2 * N + 1)

J = 0
for i in range(2 * N + 1):
    c0_des = XC_sol[:, i] - np.tile(XR_sol[:, i], (1, 4))
    c0_actual = ca.reshape(forkin_feet(XQ[:, i]), 1, 12)
    J += 10 * ca.dot(c0_des - c0_actual, c0_des - c0_actual)

    opti_kin.subject_to(opti_kin.bounded(q_bounds[:, 0], XQ[:, i], q_bounds[:, 1]))

opti_kin.minimize(J)

opti_kin.set_initial(XQ, np.tile(q_guess[:, None], 2 * N + 1))

# solve NLP
p_opts = {}
s_opts = {'print_level': 5}
opti_kin.solver('ipopt', p_opts, s_opts)
sol = opti_kin.solve()

# extract NLP output as numpy array
XQ_sol = np.array(sol.value(XQ))

# animate ##############################################################################
axis_lim = 1.5
F_len = 1 / np.max(UF_sol)

for i in range(2 * N + 1):
    r_i = np.array(XR_sol[:nr, i])
    p_feet_i = XC_sol[:, i].reshape((3, nj), order='F')
    p_leg_i = np.array(r_i + forkin_leg(XQ_sol[:, i]))
    F_i = np.array(UF_sol[:, i])

    p_i = {}
    p_i['r'] = r_i[:, None]
    p_i['r1'] = p_leg_i[:, 0][:, None]
    p_i['r3'] = p_leg_i[:, 1][:, None]
    p_i['r4'] = p_leg_i[:, 2][:, None]
    p_i['l1'] = p_leg_i[:, 3][:, None]
    p_i['l3'] = p_leg_i[:, 4][:, None]
    p_i['l4'] = p_leg_i[:, 5][:, None]
    p_i['rc1'] = p_feet_i[:, 0][:, None]
    p_i['rc2'] = p_feet_i[:, 1][:, None]
    p_i['lc1'] = p_feet_i[:, 2][:, None]
    p_i['lc2'] = p_feet_i[:, 3][:, None]

    F_vec_i = {}
    F_vec_i['rc1'] = p_i['rc1'] + F_len * F_i[0:3][:, None]
    F_vec_i['rc2'] = p_i['rc2'] + F_len * F_i[3:6][:, None]
    F_vec_i['lc1'] = p_i['lc1'] + F_len * F_i[6:9][:, None]
    F_vec_i['lc2'] = p_i['lc2'] + F_len * F_i[9:12][:, None]

    if i == 0:
        p = p_i
        F_vec = F_vec_i
    else:
        for k, v in p_i.items():
            p[k] = np.hstack((p[k], v))
        for k, v in F_vec_i.items():
            F_vec[k] = np.hstack((F_vec[k], v))

leg_r_coord = np.zeros((3, 3, 2 * N + 1))  # (cartesian space)*(# datapoints)*(# time points)
leg_l_coord = np.zeros((3, 3, 2 * N + 1))
foot_r_coord = np.zeros((3, 2, 2 * N + 1))
foot_l_coord = np.zeros((3, 2, 2 * N + 1))
torso_coord = np.zeros((3, 4, 2 * N + 1))
F_coord = np.zeros((nj, 3, 2, 2 * N + 1))  # (# forces)*(cartesian space)*(# datapoints)*(# time points)
for xyz in range(3):
    leg_r_coord[xyz, :, :] = np.array([p['r1'][xyz, :], p['r3'][xyz, :], p['r4'][xyz, :]])
    leg_l_coord[xyz, :, :] = np.array([p['l1'][xyz, :], p['l3'][xyz, :], p['l4'][xyz, :]])
    foot_r_coord[xyz, :, :] = np.array([p['rc1'][xyz, :], p['rc2'][xyz, :]])
    foot_l_coord[xyz, :, :] = np.array([p['lc1'][xyz, :], p['lc2'][xyz, :]])
    torso_coord[xyz, :, :] = np.array([p['r'][xyz, :], p['r1'][xyz, :],
                                       p['l1'][xyz, :], p['r'][xyz, :]])
    for j, key in enumerate(['rc1', 'rc2', 'lc1', 'lc2']):
        F_coord[j, xyz, :, :] = np.array([p[key][xyz, :], F_vec[key][xyz, :]])

anim_fig = plt.figure(figsize=(12, 12))
ax = Axes3D(anim_fig)
lines = [plt.plot([], [])[0] for _ in range(6 + 2 * nj)]


def animate(i):
    for j in range(nj):
        lines[j].set_data(c_des[3 * j, i], c_des[3 * j + 1, i])
        lines[j].set_3d_properties(c_des[3 * j + 2, i])

    lines[nj].set_data(xr_des[0, i], xr_des[1, i])
    lines[nj].set_3d_properties(xr_des[2, i])

    lines[nj + 1].set_data(foot_r_coord[0, :, i], foot_r_coord[1, :, i])
    lines[nj + 1].set_3d_properties(foot_r_coord[2, :, i])
    lines[nj + 2].set_data(foot_l_coord[0, :, i], foot_l_coord[1, :, i])
    lines[nj + 2].set_3d_properties(foot_l_coord[2, :, i])

    lines[nj + 3].set_data(torso_coord[0, :, i], torso_coord[1, :, i])
    lines[nj + 3].set_3d_properties(torso_coord[2, :, i])

    lines[nj + 4].set_data(leg_r_coord[0, :, i], leg_r_coord[1, :, i])
    lines[nj + 4].set_3d_properties(leg_r_coord[2, :, i])
    lines[nj + 5].set_data(leg_l_coord[0, :, i], leg_l_coord[1, :, i])
    lines[nj + 5].set_3d_properties(leg_l_coord[2, :, i])

    for j in range(nj):
        lines[nj + 6 + j].set_data(F_coord[j, 0, :, i], F_coord[j, 1, :, i])
        lines[nj + 6 + j].set_3d_properties(F_coord[j, 2, :, i])

    ax.view_init(elev=0, azim=i)
    ax.set_xlim3d([torso_coord[0, 1, i] - 1, torso_coord[0, 1, i] + 1])
    ax.set_ylim3d([torso_coord[1, 1, i] - 1, torso_coord[1, 1, i] + 1])

    return lines


ax.set_zlim3d([0, 2])
# ax.view_init(elev=0, azim=90)
# fig_scale = 1.5
# fig_shift = np.array([1.5, 1.5, -1.5])
# ax.set_xlim3d([fig_shift[0]-fig_scale, fig_shift[0]+fig_scale])
# ax.set_ylim3d([fig_shift[1]-fig_scale, fig_shift[1]+fig_scale])
# ax.set_zlim3d([fig_shift[2]-fig_scale, fig_shift[2]+fig_scale])

for line in lines[:nj]:
    line.set_color('c')
    line.set_marker('o')
    line.set_markeredgewidth(7)

lines[nj].set_color('c')
lines[nj].set_marker('o')
lines[nj].set_markeredgewidth(7)

for line in lines[nj + 1:nj + 3]:
    line.set_color('g')
    line.set_linewidth(5)
    line.set_marker('o')
    line.set_markeredgewidth(7)

for line in lines[nj + 3:nj + 6]:
    line.set_color('b')
    line.set_linewidth(5)
    line.set_marker('o')
    line.set_markeredgewidth(7)

for line in lines[nj + 6:]:
    line.set_color('r')
    line.set_linewidth(3)

# create animation
anim = animation.FuncAnimation(anim_fig, animate, frames=2 * N + 1,
                               interval=tf * 1000 / (2 * N + 1), repeat=True, blit=False)

'''
# uncomment to write to file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=int((2*N+1)/tf), metadata=dict(artist='Me'), bitrate=1000)
anim.save('point_mass_diff_foot_angle' + '.mp4', writer=writer)
'''

plt.show()
"""