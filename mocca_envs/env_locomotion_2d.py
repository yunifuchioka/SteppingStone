import gym
import numpy as np

from mocca_envs.bullet_objects import VSphere
from mocca_envs.env_base import EnvBase
from mocca_envs.robots import Crab2D, Walker2D


class Walker2DCustomEnv(EnvBase):

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    robot_class = Walker2D
    termination_height = 0.5
    robot_random_start = True
    robot_init_position = [0, 0, 1.2]
    robot_init_velocity = None

    def __init__(self, **kwargs):
        # initialize simulator, robot, and ground
        super().__init__(self.robot_class, **kwargs)

        # reference tracking reward weights
        self.body_vel_track_weight = 0.1
        self.body_rpy_track_weight = 0.1
        self.feet_rel_track_weight = 0.8

        # reference trajectories for tracking. Values are assumed to have been pre-interpolated to have
        # the same time frequency as this environment, ie 1/self.control_step Hz
        # axis 0 is time, and the remaining dimensions match the dimension of the corresponding variable
        # TODO: replace lines below with loading externally calculated trajectories,
        #  eg. by trajectory optimization
        self.traj_len = 1200
        time = np.linspace(0, self.control_step*self.traj_len, self.traj_len)
        self.body_vel_target = np.zeros((self.traj_len, 3))
        self.body_rpy_target = np.zeros((self.traj_len, 3))
        self.feet_rel_target = np.stack((
            #0 * time, 0 * time, -0.7 * np.ones(self.traj_len),
            #0 * time, 0 * time, -1 * np.ones(self.traj_len)
            0*time, 0*time, -1 + 0.3*np.maximum(np.sin(time), 0),
            0*time, 0*time, -1 + 0.3*np.maximum(-np.sin(time), 0)
        ), axis=1).reshape(self.traj_len, 2, 3)

        # Observation space is the augmented state, see calc_aug_state()
        self.reset() # must be called for self.aug_state to be initialized properly
        high = np.inf * np.ones(self.aug_state.shape[0])
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        # Action space is for the robot
        self.action_space = self.robot.action_space

    def reset(self):
        self.done = False

        self.traj_idx = 0

        # restoreState resets simulation to a saved checkpoint
        # including robot position and pose, and all other objects
        if self.state_id >= 0:
            self._p.restoreState(self.state_id)

        # set robot's initial pose
        # randomness can help with local minima
        self.robot_state = self.robot.reset(
            random_pose=self.robot_random_start,
            pos=self.robot_init_position,
            vel=self.robot_init_velocity,
        )

        # calculate the augmented state to be input into the network
        self.calc_aug_state()

        # make camera point to robot's root joint
        if self.is_rendered or self.use_egl:
            self.camera.lookat(self.robot.body_xyz)

        # save a checkpoint after first environment reset
        if not self.state_id >= 0:
            self.state_id = self._p.saveState()

        # environment's observation space is robot's state plus additional variables to aid in training
        # see self.calc_aug_state() for what's included in the observation
        return self.aug_state

    def step(self, action):
        # set torque for each joint
        # action is normalized to between [-1, 1], wrt max. torque for each joint
        # control frequency is 60Hz, i.e. everytime this function is called, 1/60 seconds has passed in simulation
        self.robot.apply_action(action)

        # step simulation forward after torque is set
        # simulation period is `control_step / sim_frame_skip`, or 240Hz in this case
        # increase sim_frame_skip to make simulation more stable, if necessary
        self.scene.global_step()

        # recalculate robot state after simulation has stepped forward
        # ground_ids is passed to calculate contact state
        self.robot_state = self.robot.calc_state(self.ground_ids)

        # calculate rewards and if episode should be terminated
        self.calc_env_state(action)

        # calculate the augmented state to be input into the network
        self.calc_aug_state()

        #reward = self.track_reward * self.joint_limit_reward * self.energy_reward
        reward = self.track_reward * self.joint_limit_reward

        # for rendering only, in the pybullet gui, press
        # <space> to pause, 'r' to reset, etc
        if self.is_rendered or self.use_egl:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)

        # step the reference trajectory index forward, after all simulation, state, and reward calculations
        if self.traj_idx < self.traj_len-1:
            self.traj_idx += 1

        # step() should return observation, reward, done (boolean), and info (dict)
        # info can be anything, some people return individual reward components
        # like, {"progress": self.progress, "energy": self.energy_penalty, ...}
        return self.aug_state, reward, self.done, {}

    def calc_track_rewards(self):
        body_vel_error = self.robot.body_vel - self.body_vel_target[self.traj_idx]
        body_rpy_error = self.robot.body_rpy - self.body_rpy_target[self.traj_idx]
        feet_rel_error = (self.robot.feet_xyz - self.robot.body_xyz) - self.feet_rel_target[self.traj_idx]

        self.body_vel_reward = np.exp( -1.0 * (body_vel_error**2).sum() )
        self.body_rpy_reward = np.exp( -30.0 * (body_rpy_error**2).sum() )
        self.feet_rel_reward = np.exp( -1.0 * (feet_rel_error**2).sum() )

    def calc_aug_state(self):
        # calculates the augmented state that is to be fed into the network
        feet_rel_flat = (self.robot.feet_xyz - self.robot.body_xyz).flatten()
        feet_rel_target_flat = self.feet_rel_target[self.traj_idx].flatten()
        self.aug_state = np.concatenate((
            self.robot_state,
            self.robot.body_vel,
            self.robot.body_rpy,
            feet_rel_flat,
            self.body_vel_target[self.traj_idx],
            self.body_rpy_target[self.traj_idx],
            feet_rel_target_flat
        ))

    def calc_env_state(self, action):
        # in case if neural net explodes
        if not np.isfinite(self.robot_state).all():
            print("~INF~", self.robot_state)
            self.done = True

        # calculate rewards
        self.calc_track_rewards()
        self.track_reward = \
            + self.body_vel_track_weight * self.body_vel_reward \
            + self.body_rpy_track_weight * self.body_rpy_reward \
            + self.feet_rel_track_weight * self.feet_rel_reward

        energy_used = np.abs(action * self.robot.joint_speeds).mean()
        self.energy_reward = np.exp( -10.0 * (energy_used**2).sum() )
        self.joint_limit_reward = np.exp( -0.05 * self.robot.joints_at_limit**2 )

        # tall bonus encourages robot to stay upright
        # note1: roll and pitch, corresponding to self.robot_state[4:6], are defined so that
        #   pitch is always within (-pi/2, pi/2) and roll has discrete values -pi, 0, pi
        #   depending on whether the biped is "bent over" or not (0 if not bent over)
        # note2: self.robot_state[4:6] and self.robot.body_rpy[0:2] have the same values up to
        #   numerical differences due to np.float32 casting
        if (self.robot_state[0] > self.termination_height
                and np.abs(self.robot_state[4]) < np.pi):
            self.tall_bonus = 1.0
        else:
            self.tall_bonus = -1.0

        # since getting up can be hard to learn
        # terminate the episode when robot falls
        self.done = self.done or self.tall_bonus < 0


class Crab2DCustomEnv(Walker2DCustomEnv):

    # same implementation as Walker2D
    # only changing the robot class and robot specific settings
    robot_class = Crab2D
    termination_height = 0.5
    robot_init_position = [0, 0, 1.2]