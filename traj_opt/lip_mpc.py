"""
Stand-alone trajectory optimization script that solves the LIP-ZMP tracking problem
described in Kajita et al 2003 "Biped walking pattern generation by using preview control 
of zero-moment point", although through trajectory optimization and not through preview
control
"""

import itertools

import numpy as np
import casadi as ca
from scipy.interpolate import CubicSpline, CubicHermiteSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import plotly.graph_objects as go
import plotly.express as px
import operator
from collections import namedtuple
from scipy import signal
import functools
import itertools

Footstep = namedtuple('Footstep', ['t', 'x', 'y'])

def footstep_generator(random=True):
    #stride_duration = 0.8
    stride_duration = 0.5
    #stride_length = 0.2
    #stride_length = 0.4
    stride_length = 0.7
    stride_width = 0.3
    t = 0
    x = 0
    side = -1
    yield Footstep(t, x, 0)
    t += stride_duration
    yield Footstep(t, x, stride_width / 2)
    while True:
        t += stride_duration
        x += np.random.normal(loc=stride_length, scale=0.05 if random else 0)
        y = np.random.normal(loc = side * stride_width/2, scale=0.03 if random else 0)
        side *= -1
        yield Footstep(t, x, y)

stride_height = 0.2

def footstep_zmp(footsteps):
    t, x, y = map(list, zip(*footsteps))
    t = [t[0] - 1] + t + [t[-1] + 1]
    x = [x[0]] + x + [x[-1]]
    y = [y[0]] + y + [x[-1]]
    dy_dx = np.zeros((2, len(x)))
    return CubicHermiteSpline(t, np.vstack([x,y]), dy_dx, axis=1, extrapolate=True)

class LipOpti():
    def __init__(self, solution_duration=2, dt=0.1):
        # com height
        zc = 0.9 # height of thigh + calf in pybullet walker2D model
        self.com_height = zc
        # x, y dimensions of feet
        foot_size = np.array([0.25, 0.15])
        # y-width of torso, z-height of torso (from hip joint to COM)
        torso_size = np.array([0.3, 0.5])
        # length of thigh and calves respectively
        leg_length = np.array([0.5, 0.5])

        nx = 6 # dimension of state vector
        nu = 2 # dimension of control vector
        ny = 2 # dimension of output vector
        g = 9.81 # gravity
        com_accel_max = 10.0 # maximum COM acceleration
        Q = np.array([500, 500]) # output error cost weight
        Qf = np.array([50, 50]) #final output error cost weight
        R = np.array([0.1, 0.1]) #input regulation cost weight

        self.N = int(np.ceil(solution_duration / dt))
        N = self.N
        self.t_f = self.N * dt
        self.t = np.linspace(0, self.t_f, 2*N + 1)

        self.opti = ca.Opti()

        # decision variables
        X = self.opti.variable(nx, 2*N+1)
        self.X = X
        U = self.opti.variable(nu, 2*N+1)
        self.U = U

        # parameters
        self.x_init = self.opti.parameter(nx)
        self.y_des = self.opti.parameter(ny, 2*N+1)

        # dynamics
        # symbolic variables used to derive equations of motion
        x = ca.SX.sym('x', nx) #state
        u = ca.SX.sym('u', nu) #control
        # eq 12 in Kajita et al 2003, expanded to include both x and y components
        A = ca.SX(np.array([ \
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]))
        B = ca.SX(np.array([ \
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            ]))
        C = ca.SX(np.array([ \
            [1.0, 0.0, 0.0, 0.0, -zc/g, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, -zc/g]
            ]))
        f = ca.Function('f', [x,u], [A@x+B@u])
        self.y = ca.Function('y', [x], [C@x])
        y = self.y

        # objective
        # simpson quadrature coefficients, to be used to compute integrals
        simp = np.empty((1,2*N+1))
        simp[0,::2] = 2
        simp[0,1::2] = 4
        simp[0,0], simp[0,-1]  = 1, 1

        # cost = finite horizon LQR cost computed with simpson quadrature
        # here, regulation is performed on the COM accel and not input
        J = 0.0
        for k in range(2):
            J += R[k] * ca.dot(simp, X[4+k,:]*X[4+k,:])

        for j in range(ny):
            for i in range(2*N+1):
                J += Q[j] * simp[0][i] \
                    * (y(X[:,i])[j]-self.y_des[j,i])*(y(X[:,i])[j]-self.y_des[j,i])
            J += Qf[j] * (y(X[:,-1])[j]-self.y_des[j,-1])*(y(X[:,-1])[j]-self.y_des[j,-1])

        self.opti.minimize(J)

        # COM accel constraint
        self.opti.subject_to(self.opti.bounded( \
            np.full(X[4:,:].shape, -com_accel_max), 
            X[4:,:], 
            np.full(X[4:,:].shape, com_accel_max)
            ))

        # initial condition constraint
        self.opti.subject_to(X[:,0] == self.x_init)

        for i in range(2*N+1):
            if i%2 != 0:
                # for each finite element:
                x_left, x_mid, x_right = X[:,i-1], X[:,i], X[:,i+1]
                u_left, u_mid, u_right = U[:,i-1], U[:,i], U[:,i+1]
                f_left, f_mid, f_right = f(x_left,u_left), f(x_mid,u_mid), f(x_right,u_right)

                # interpolation constraints
                self.opti.subject_to( \
                    # equation (6.11) in Kelly 2017
                    x_mid == (x_left+x_right)/2.0 + self.t_f/N*(f_left-f_right)/8.0)

                # collocation constraints
                self.opti.subject_to( \
                    # equation (6.12) in Kelly 2017 
                    self.t_f/N*(f_left+4*f_mid+f_right)/6.0 == x_right-x_left)

        p_opts = {}
        s_opts = {'print_level': 5}
        self.opti.solver('ipopt', p_opts, s_opts)
        
    def get_solution(self, t_i, x_i, footsteps, dt=1/60, plot=False):
        y_des_t = footstep_zmp(footsteps)
        y_des = y_des_t(self.t + t_i)
        self.opti.set_value(self.x_init, x_i)
        self.opti.set_value(self.y_des, y_des)
        self.opti.set_initial(self.X, np.vstack((
            y_des,
            y_des_t(self.t + t_i, 1),
            y_des_t(self.t + t_i, 2),
        )))
        self.opti.set_initial(self.U, np.zeros(self.U.shape))
        solution = self.opti.solve()
        sol_x = np.array(solution.value(self.X))
        sol_interpolator = CubicSpline(self.t, sol_x, axis=1)
        sol_y = np.array(solution.value(self.y(sol_x)))
        if not plot:
            return sol_interpolator(np.arange(0, self.t_f, dt))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x = [f.x for f in footsteps],
            y = [f.y for f in footsteps],
            z = [0] * len(footsteps),
        ))
        fig.add_trace(go.Scatter3d(
            x = y_des[0],
            y = y_des[1],
            z = [0] * len(y_des[0]),
            text = self.t + t_i,
            marker_size=5,
            mode="markers",
        ))
        fig.add_trace(go.Scatter3d(
            x = sol_y[0],
            y = sol_y[1],
            z = [0] * len(y_des[0]),
            text = self.t + t_i,
        ))
        fig.add_trace(go.Scatter3d(
            x = sol_x[0],
            y = sol_x[1],
            z = [self.com_height] * len(y_des[0]),
            text = self.t + t_i,
        ))
        fig.show()

lip_opti = LipOpti(solution_duration=8)
footsteps = list(itertools.islice(footstep_generator(random=False), 20))
#lip_opti.get_solution(0, np.array([footsteps[0].x, footsteps[0].y, 0,0,0,0]), footsteps, plot=True)

# modifications to generate trajectories offline
from scipy.interpolate import interp1d

lip_sol = lip_opti.get_solution(0, np.array([footsteps[0].x, footsteps[0].y, 0,0,0,0]), footsteps, plot=False)

# discrete footstep positions at foot touchdown
footstep_l_points = np.array([[footstep.t, footstep.x] for footstep in footsteps if footstep.y>=0]).T
footstep_r_points = np.array([[footstep.t, footstep.x] for footstep in footsteps if footstep.y<=0]).T

# footstep interpolators using zero order hold
left_interp = interp1d(footstep_l_points[0], footstep_l_points[1], kind='zero')
right_interp = interp1d(footstep_r_points[0], footstep_r_points[1], kind='zero')

# footstep trajectory
tfine = np.linspace(0,8,8*60)
left_traj = left_interp(tfine)
right_traj = right_interp(tfine)

"""
# check what trajectory looks like
plt.plot(tfine, left_traj)
plt.plot(tfine, right_traj)
plt.plot(tfine, lip_sol[0,:])
plt.show()
"""

lip_traj = np.vstack((left_traj, right_traj, lip_sol[0,:]))

#import ipdb; ipdb.set_trace()
np.save('../trajectories/lip_traj4.npy', lip_traj)

exit()
