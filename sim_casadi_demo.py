# %%
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import meshcat.animation as anim

# %%

class Renderer():

    def __init__(self, params):
        self.params = params
        self.table_l = params['table_l']
        self.table_w = params['table_w']    
        self.table_h = params['table_h']
        self.net_l = params['net_l']
        self.net_w = params['net_w']
        self.net_h = params['net_h']
        self.ball_r = params['ball_r']
        self.racket_r = params['racket_r']
        self.racket_t = params['racket_t']
        self.racket_handle_l = 0.1
        self.racket_handle_w = 0.03

        self.m_t = params['m_t']
        self.m_l2 = params['m_l2']
        self.m_l3 = params['m_l3']

        self.traj_history = []
        # self.render_init()


    def render_init(self, fk_func=None):
        # self.vis = meshcat.Visualizer().open()
        self.vis = meshcat.Visualizer()
        # self.vis["/Scene"].set_property("background", [0, 0, 0])
        # self.vis["/Background"].set_property("top_color", [0, 0, 0])
        # self.vis["/Background"].set_property("bottom_color", [0, 0, 0])
        self.vis["/Grid"].set_property("visible", False)

        # self.vis = meshcat.Visualizer()
        self.vis['table'].set_object(g.Box([self.table_l, self.table_w, self.table_h]))
        self.vis['table'].set_transform(tf.translation_matrix([0,0,-self.table_h/2]))  
        self.vis['table'].set_property("color", [70/255, 160/255, 126/255, 1]) 
        
        self.vis['net'].set_object(g.Box([self.net_l,self.net_w,self.net_h]))
        self.vis['net'].set_transform(tf.translation_matrix([0,0,self.net_h/2]))  
        self.vis['net'].set_property("color", [0/255, 0/255, 0/255, 1])
        
        self.vis['ball'].set_object(g.Sphere(self.ball_r))
        self.vis['ball'].set_transform(tf.translation_matrix([0,0,self.ball_r]))  
        self.vis['ball'].set_property("color", [255/255, 255/255, 255/255, 1]) 

        axis_length = 0.1
        self.vis['ball/axis/x'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [axis_length, 0, 0]]).T), g.MeshBasicMaterial(color=0xff0000)))
        self.vis['ball/axis/y'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [0, axis_length, 0]]).T), g.MeshBasicMaterial(color=0x00ff00)))
        self.vis['ball/axis/z'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [0, 0, axis_length]]).T), g.MeshBasicMaterial(color=0x0000ff)))

        self.vis['racket1'].set_object(g.Box([self.racket_r*2, self.racket_r*2, self.racket_t]))
        self.vis['racket1'].set_property("color", [255/255, 0/255, 0/255, 0.1])
        self.vis['racket1/handle'].set_object(g.Box([self.racket_handle_w, self.racket_handle_l, self.racket_t]))
        self.vis['racket1/handle'].set_transform(tf.translation_matrix([0,self.racket_handle_l/2 + self.racket_r ,0]))  
        self.vis['racket1/handle'].set_property("color", [255/255, 0/255, 0/255, 0.1])
        self.vis['racket1'].set_transform(tf.translation_matrix([-self.table_l/2-0.5,0,self.net_h/2]))  
        self.vis['racket1/axis/x'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [axis_length, 0, 0]]).T), g.MeshBasicMaterial(color=0xff0000)))
        self.vis['racket1/axis/y'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [0, axis_length, 0]]).T), g.MeshBasicMaterial(color=0x00ff00)))
        self.vis['racket1/axis/z'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [0, 0, axis_length]]).T), g.MeshBasicMaterial(color=0x0000ff)))

        self.vis['racket2'].set_object(g.Box([self.racket_r*2, self.racket_r*2, self.racket_t]))
        self.vis['racket2'].set_property("color", [0/255, 0/255, 255/255, 0.1])
        self.vis['racket2/handle'].set_object(g.Box([self.racket_handle_w, self.racket_handle_l, self.racket_t]))
        self.vis['racket2/handle'].set_transform(tf.translation_matrix([0,self.racket_handle_l/2 + self.racket_r ,0]))  
        self.vis['racket2/handle'].set_property("color", [0/255, 0/255, 255/255, 0.1])
        self.vis['racket2'].set_transform(tf.translation_matrix([+self.table_l/2+0.5,0,self.net_h/2]))  
        self.vis['racket2/axis/x'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [axis_length, 0, 0]]).T), g.MeshBasicMaterial(color=0xff0000)))
        self.vis['racket2/axis/y'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [0, axis_length, 0]]).T), g.MeshBasicMaterial(color=0x00ff00)))
        self.vis['racket2/axis/z'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [0, 0, axis_length]]).T), g.MeshBasicMaterial(color=0x0000ff)))


        self.vis['m1_base'].set_object(g.Box([self.m_t, self.m_t, self.m_t]))
        self.vis['m1_base'].set_property("color", [70/255, 160/255, 126/255, 1])
        self.vis['m1_l1'].set_object(g.Box([self.m_t, self.m_t, self.m_t]))
        self.vis['m1_l1'].set_property("color", [0.5,0.5,0.5,1])
        self.vis['m1_l2'].set_object(g.Box([self.m_t, self.m_t, self.m_l2]))
        self.vis['m1_l2'].set_property("color", [1,1,1,1])
        self.vis['m1_l3'].set_object(g.Box([self.m_t, self.m_t, self.m_l3]))
        self.vis['m1_l3'].set_property("color", [0.5,0.5,0.5,1])
        self.vis['m1_l4'].set_object(g.Box([self.m_t, self.m_t, self.m_t]))
        self.vis['m1_l4'].set_property("color", [1,1,1,1])
        self.vis['m1_l5'].set_object(g.Box([self.m_t, self.m_t, self.m_t]))
        self.vis['m1_l5'].set_property("color", [0.5,0.5,0.5,1])
        self.vis['m1_l6'].set_object(g.Box([self.racket_r*2, self.racket_r*2, self.racket_t]))
        self.vis['m1_l6'].set_property("color", [1,0,0,0.5])
        self.vis['m1_l6/axis/x'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [axis_length, 0, 0]]).T), g.MeshBasicMaterial(color=0xff0000)))
        self.vis['m1_l6/axis/y'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [0, axis_length, 0]]).T), g.MeshBasicMaterial(color=0x00ff00)))
        self.vis['m1_l6/axis/z'].set_object(g.Line(g.PointsGeometry(np.array([[0, 0, 0], [0, 0, axis_length]]).T), g.MeshBasicMaterial(color=0x0000ff)))

        # initially rendered at zero joint angles
        gs = fk_func(np.zeros((12,1))) # returns every transformation
        gs = [i.toarray() for i in gs]
        links = ['m1_base', 'm1_l1', 'm1_l2', 'm1_l3', 'm1_l4', 'm1_l5', 'm1_l6']
        for i, gsi in enumerate(gs):
            self.vis[links[i]].set_transform(gsi)


    def reset(self):
        self.traj_history = []
        self.vis['traj'].set_object(g.Line(g.PointsGeometry(np.array([]).T)))

    def render(self, ball_x, racket1_x=None, racket2_x=None, traj=False):

        self.vis['ball'].set_transform(tf.translation_matrix(ball_x[0:3]) @ tf.euler_matrix(*ball_x[3:6])) 
            
        if traj==True:
            self.traj_history.append(ball_x[0:3])
            self.vis['traj'].set_object(g.Line(g.PointsGeometry(np.array(self.traj_history).T)))   
        else:
            self.vis['traj'].set_object(g.Line(g.PointsGeometry(np.array([]).T)))

        if isinstance(racket1_x, np.ndarray):
            self.vis['racket1'].set_transform(tf.translation_matrix(racket1_x[0:3]) @ tf.euler_matrix(*racket1_x[3:6]))
        if isinstance(racket2_x, np.ndarray):
            self.vis['racket2'].set_transform(tf.translation_matrix(racket2_x[0:3]) @ tf.euler_matrix(*racket2_x[3:6]))


    def render_anim(self, ts, ball_xs, racket1_xs=None, racket2_xs=None, m1_xs=None, fk_func=None,  traj=False):
        self.reset()
        self.anim = anim.Animation(default_framerate=1)
        if traj==True:
            self.vis['traj'].set_object(g.Line(g.PointsGeometry(ball_xs[0:3,:])))
        for i in range(ball_xs.shape[1]):
            with self.anim.at_frame(self.vis, ts[i]) as frame:
                frame['ball'].set_transform(tf.translation_matrix(ball_xs[0:3, i]) @ tf.euler_matrix(*ball_xs[3:6, i]))
                if isinstance(racket1_xs, np.ndarray):
                    frame['racket1'].set_transform(tf.translation_matrix(racket1_xs[0:3, i]) @ tf.euler_matrix(*racket1_xs[3:6, i]))
                if isinstance(racket2_xs, np.ndarray):
                    frame['racket2'].set_transform(tf.translation_matrix(racket2_xs[0:3, i]) @ tf.euler_matrix(*racket2_xs[3:6, i]))
                if isinstance(m1_xs, np.ndarray):
                    gs = fk_func(m1_xs[:12,i])
                    gs = [i.toarray() for i in gs]
                    links = ['m1_base', 'm1_l1', 'm1_l2', 'm1_l3', 'm1_l4', 'm1_l5', 'm1_l6']
                    for i, gsi in enumerate(gs):
                        frame[links[i]].set_transform(gsi)


        self.vis.set_animation(self.anim)


# %% ball_traj_stuff

def rk4(params, dynamics_func, x, u, dt):
    k1 = dt * dynamics_func(params, x, u)
    k2 = dt * dynamics_func(params, x + k1/2, u)
    k3 = dt * dynamics_func(params, x + k2/2, u)
    k4 = dt * dynamics_func(params, x + k3, u)
    return x + (1/6) * (k1 + 2*k2 + 2*k3 + k4) 


def ball_flight_dynamics(params, ball_x, ball_u):
    g, C_D, C_L = params['g'], params['C_D'], params['C_L']
    ball_pos, ball_rot, ball_vel, ball_ang_vel = ball_x[0:3], ball_x[3:6], ball_x[6:9], ball_x[9:12]
    dx = ca.vertcat(ball_vel,
                    ball_ang_vel,
                    g - C_D*ca.norm_2(ball_vel)*ball_vel + C_L*ca.cross(ball_ang_vel, ball_vel),
                    [0,0,0],)
    return dx

def table_contact_reset_map(params, ball_x_in, ball_u_in, ball_x_last):

    ball_r = params['ball_r']
    mu = params['mu']  
    epsilon_t = params['epsilon_t']  


    tangential_vel = ca.vertcat(ball_x_in[6] - ball_r * ball_x_in[10], 
                                ball_x_in[7] + ball_r * ball_x_in[9], 
                                0)
    alpha = mu * (1 + epsilon_t) *  ca.fabs(ball_x_in[8]) / ca.norm_2(tangential_vel)
    
    A_v = ca.if_else(alpha < 0.4, 
                     ca.diag(ca.vertcat(1 - alpha, 1 - alpha, -epsilon_t)), 
                     ca.diag(ca.vertcat(3/5, 3/5, -epsilon_t)))
    
    T = type(ca.vertcat(ball_x_in)) # to infer if input is np/MX to creates 0's of DM/MX
    
    # B_v = ca.MX.zeros(3, 3)
    B_v = T.zeros(3,3)
    B_v[0, 1] = ca.if_else(alpha < 0.4, alpha * ball_r, 2 * ball_r / 5)
    B_v[1, 0] = ca.if_else(alpha < 0.4, -alpha * ball_r, -2 * ball_r / 5)

    # A_w = ca.MX.zeros(3, 3)
    A_w = T.zeros(3,3)
    A_w[0, 1] = ca.if_else(alpha < 0.4, -3 * alpha / (2 * ball_r), -3 / (5 * ball_r))
    A_w[1, 0] = ca.if_else(alpha < 0.4, 3 * alpha / (2 * ball_r), 3 / (5 * ball_r))

    B_w = ca.if_else(alpha < 0.4, 
                     ca.diag(ca.vertcat(1 - 3 * alpha / 2, 1 - 3 * alpha / 2, 1)), 
                     ca.diag(ca.vertcat(2/5, 2/5, 1)))

    # ball_x_out = T(ball_x_in) 
    # ball_x_out[0:3] = ball_x_last[0:3]
    # ball_x_out[3:6] = ball_x_in[3:6]
    # ball_x_out[6:9] = A_v @ ball_x_in[6:9] + B_v @ ball_x_in[9:12]
    # ball_x_out[9:12] = A_w @ ball_x_in[6:9] + B_w @ ball_x_in[9:12]

    ball_x_out = ca.vertcat(ball_x_last[0:3],
                            ball_x_last[3:6],
                            A_v @ ball_x_in[6:9] + B_v @ ball_x_in[9:12],
                            A_w @ ball_x_in[6:9] + B_w @ ball_x_in[9:12],)
    return ball_x_out

def check_table_contact(params, ball_x):
    condition_x = ca.fabs(ball_x[0]) <= params['table_l'] / 2
    condition_y = ca.fabs(ball_x[1]) <= params['table_w'] / 2
    condition_z = ball_x[2] <= params['ball_r']
    return ca.logic_and(condition_x, ca.logic_and(condition_y, condition_z))

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Returns the 3x3 rotation matrix from ZYX Euler angles (roll, pitch, yaw).
    roll  = rotation about x-axis
    pitch = rotation about y-axis
    yaw   = rotation about z-axis
    """

    cr = ca.cos(roll)
    sr = ca.sin(roll)
    cp = ca.cos(pitch)
    sp = ca.sin(pitch)
    cy = ca.cos(yaw)
    sy = ca.sin(yaw)

    R = ca.vertcat(
        ca.horzcat(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
        ca.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
        ca.horzcat(-sp,    cp*sr,           cp*cr)
    )
    return R

def racket_contact_reset_map(params, ball_x_in, racket_x, ball_x_last):

    ball_r = params['ball_r']
    k_pv = params['k_pv']  
    k_pw = params['k_pw'] 
    epsilon_r = params['epsilon_r'] 

    T = type(ca.vertcat(ball_x_in))

    R = euler_to_rotation_matrix(racket_x[3], racket_x[4], racket_x[5])

    A_v = ca.diag(ca.vertcat(1 - k_pv, 1 - k_pv, -epsilon_r))
    B_v = T.zeros(3,3)
    B_v[0,1] = k_pv*ball_r
    B_v[1,0] = k_pv*(-ball_r)
    A_v_bar = R @ A_v @ R.T

    A_w = T.zeros(3,3)
    A_w[0,1] = k_pw*(-ball_r)
    A_w[1,0] = k_pw*ball_r
    B_w = ca.diag(ca.vertcat(1 - k_pw*(ball_r**2), 1 - k_pw*(ball_r**2), 1))
    A_w_bar = R @ A_w @ R.T

    ball_x_out = ca.vertcat(ball_x_last[0:3],
                            ball_x_last[3:6],
                            (T.eye(3) - A_v_bar) @ racket_x[6:9] + A_v_bar @ ball_x_in[6:9] + R @ B_v @ R.T @ ball_x_in[9:12],
                            (T.eye(3) - A_w_bar) @ racket_x[6:9] + A_w_bar @ ball_x_in[6:9] + R @ B_w @ R.T @ ball_x_in[9:12])
    
    return ball_x_out

def check_racket_contact(params, ball_x, racket_x):

    R = euler_to_rotation_matrix(racket_x[3], racket_x[4], racket_x[5])
    diff_vector = ball_x[0:3] - racket_x[0:3]
    normal_vector = R @ ca.vertcat(0,0,1)

    condition_1 = ca.dot(diff_vector, normal_vector) <= 0
    condition_2 = ca.norm_2(diff_vector) <= params['racket_r']
    return  ca.logic_and(condition_1, condition_2)

# ca_x = ca.MX.sym('test',params['nx'])
# np_x = np.ones(12)
# print(f'{type(ball_flight_dynamics(params, ca_x, None))=}')
# print(f'{type(ball_flight_dynamics(params, np_x, None))=}')
# print(f'{type(table_contact_reset_map(params, ca_x, None, ca_x))=}')
# print(f'{type(table_contact_reset_map(params, np_x, None, np_x))=}')
# print(f'{type(check_table_contact(params, ca_x))=}')
# print(f'{type(check_table_contact(params, np_x))=}')
# print(f'{type(rk4(params, ball_flight_dynamics, ca_x, None, 0.01))=}')
# print(f'{type(rk4(params, ball_flight_dynamics, np_x, None, 0.01))=}')
# print(f'{type(racket_contact_reset_map(params, ca_x, ca_x, ca_x))=}')
# print(f'{type(racket_contact_reset_map(params, np_x, np_x, np_x))=}')
# print(f'{type(check_racket_contact(params, ca_x, ca_x))=}')
# print(f'{type(check_racket_contact(params, np_x, np_x))=}')


# %% manipulator_stuff

params = {
    'table_l': 2.74,
    'table_w': 1.525,
    'table_h': 0.012,

    'net_l': 0.002,
    'net_w': 1.83,
    'net_h': 0.1525,

    'ball_r': 0.02,
    'ball_m': 2.7e-3,
    'racket_r':0.08,
    'racket_t':0.001,

    'm_t':0.1,
    'm_l2':0.4,
    'm_l3':0.4,

    'mu': 0.102,
    'epsilon_t': 0.883,

    'k_p': 1.9e-3,
    'k_pv': 0.703, # k_p/m
    'k_pw': 2222.222, # k_p/I = k_p/(2/3mr^2)
    'epsilon_r': 0.788,
    # 'epsilon_r': 0.1,

    'g': np.array([0, 0, -9.8]).T,
    'C_D': 0.141,
    'C_L': 0.001,

    'nx': 12,
    'nu': 6,

    'n1': 50,
    'n2': 50,
    'n3': 50,
    'n4': 0,

    'x0': np.array([1.5,-0.5,0.25, 0,0,0, -4,0,3, 0,0,0]),
    # 'x0':np.array([ 1.5,0.,0.25,0.,0.,0.,-4.55770133,-0.67224257,2.92995921,28.27215379,86.13413194,172.90450353]),
    
    'xg': np.array([1.2,-0.5,0, 0,0,0, 0,0,0, 0,300,0]),
    'tg': 0.55,
}

def transform_matrix(x, y, z, roll, pitch, yaw):
    """Returns a 4x4 homogeneous transform matrix from RPY and translation."""
    # Rotation matrices
    Rx = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, ca.cos(roll), -ca.sin(roll)),
        ca.horzcat(0, ca.sin(roll), ca.cos(roll))
    )

    Ry = ca.vertcat(
        ca.horzcat(ca.cos(pitch), 0, ca.sin(pitch)),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-ca.sin(pitch), 0, ca.cos(pitch))
    )

    Rz = ca.vertcat(
        ca.horzcat(ca.cos(yaw), -ca.sin(yaw), 0),
        ca.horzcat(ca.sin(yaw), ca.cos(yaw), 0),
        ca.horzcat(0, 0, 1)
    )

    R = Rz @ Ry @ Rx
    p = ca.vertcat(x, y, z)
    T = ca.vertcat(
            ca.horzcat(R, p),
            ca.horzcat(0, 0, 0, 1)
        )
    return T

# def decompose_transform(T):
#     """
#     Extract x, y, z, roll, pitch, yaw from a 4x4 homogeneous transformation matrix.
#     Assumes ZYX (yaw-pitch-roll) Euler angle convention.
#     """
#     # Extract rotation matrix and translation vector
#     R = T[:3, :3]
#     x = T[0, 3]
#     y = T[1, 3]
#     z = T[2, 3]

#     # Extract yaw, pitch, roll from rotation matrix using ZYX convention
#     # R = Rz(yaw) * Ry(pitch) * Rx(roll)
#     pitch = ca.arcsin(-R[2, 0])
    
#     # Handle gimbal lock case separately
#     # If pitch is +-90 degrees (i.e. cos(pitch) ~ 0), roll and yaw are not uniquely defined
#     cos_pitch = ca.cos(pitch)
#     eps = 1e-6  # Small tolerance
#     ifabs = ca.fabs(cos_pitch) > eps

#     yaw = ca.if_else(ifabs,
#                      ca.atan2(R[1, 0], R[0, 0]),
#                      ca.atan2(-R[0, 1], R[1, 1]))
#     roll = ca.if_else(ifabs,
#                       ca.atan2(R[2, 1], R[2, 2]),
#                       0)

#     return ca.vertcat(x, y, z, roll, pitch, yaw)

def decompose_transform(T):
    """
    Extract x, y, z, roll, pitch, yaw from a 4x4 homogeneous transformation matrix.
    Assumes ZYX (yaw-pitch-roll) Euler angle convention.
    No singularity checks.
    """
    # Extract rotation matrix and translation vector
    R = T[:3, :3]
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]

    # Compute Euler angles (ZYX convention)
    # pitch = ca.arcsin(-R[2, 0])
    pitch = ca.atan2(-R[2, 0], ca.sqrt(R[0, 0]**2 + R[1, 0]**2))
    yaw = ca.atan2(R[1, 0], R[0, 0])
    roll = ca.atan2(R[2, 1], R[2, 2])

    return ca.vertcat(x, y, z, roll, pitch, yaw)

def ulta_hat(T):
    return ca.vertcat(T[0,3], T[1,3], T[2,3],-T[1,2], T[0,2], -T[0,1])

x = ca.MX.sym("x", 12)
q = x[:6]
q_dot = x[6:]
u = ca.MX.sym("u", 6)

t, l2, l3 = params['m_t'], params['m_l2'], params['m_l3']
racket_r, racket_t = params['racket_r'], params['racket_t']

gs = [transform_matrix(0,0,0,0,0,0)]*7
gs[0] = transform_matrix(-1.6,0,0,0,0,0) # world position of manipulator (fixed)
gs[1] = gs[0] @ transform_matrix(0,0,t,0,0,q[0]) 
gs[2] = gs[1] @ transform_matrix(0,0,0,0,q[1],0) @ transform_matrix(0,-t,l2/2-t/2,0,0,0)
gs[3] = gs[2] @ transform_matrix(0,0,l2/2-t/2,0,q[2],0) @ transform_matrix(0,t,l3/2-t/2,0,0,0)
gs[4] = gs[3] @ transform_matrix(0,0,l3/2-t/2,0,q[3],0) @ transform_matrix(0,-t,0,0,0,0)
gs[5] = gs[4] @ transform_matrix(0,0,0,0,0,q[4]) @ transform_matrix(0,0,t,0,0,0)
gs[6] = gs[5] @ transform_matrix(0,0,0,0,q[5],0) @ transform_matrix(0,-t/2-racket_r,0,0,0,0)

Js = [transform_matrix(0, 0, 0, 0, 0, 0)]*7
dgs1_dx = ca.jacobian(gs[1],x)
Js[1] = ca.horzcat(*[ulta_hat(ca.inv(gs[1]) @ ca.reshape(dgs1_dx[:,i], (4,4))) for i in range(6)])
dgs2_dx = ca.jacobian(gs[2],x)
Js[2] = ca.horzcat(*[ulta_hat(ca.inv(gs[2]) @ ca.reshape(dgs2_dx[:,i], (4,4))) for i in range(6)])
dgs3_dx = ca.jacobian(gs[3],x)
Js[3] = ca.horzcat(*[ulta_hat(ca.inv(gs[3]) @ ca.reshape(dgs3_dx[:,i], (4,4))) for i in range(6)])
dgs4_dx = ca.jacobian(gs[4],x)
Js[4] = ca.horzcat(*[ulta_hat(ca.inv(gs[4]) @ ca.reshape(dgs4_dx[:,i], (4,4))) for i in range(6)])
dgs5_dx = ca.jacobian(gs[5],x)
Js[5] = ca.horzcat(*[ulta_hat(ca.inv(gs[5]) @ ca.reshape(dgs5_dx[:,i], (4,4))) for i in range(6)])
dgs6_dx = ca.jacobian(gs[6],x)
Js[6] = ca.horzcat(*[ulta_hat(ca.inv(gs[6]) @ ca.reshape(dgs6_dx[:,i], (4,4))) for i in range(6)])


Ms = [None]*7

m1 = 1
i1 = (t**2)*m1/6
Ms[0] = ca.diag(ca.DM([m1,m1,m1,i1,i1,i1]))
Ms[1] = ca.diag(ca.DM([m1,m1,m1,i1,i1,i1]))
m2 = l2/t*1
i2x = i2y = (1/12)*m2*(l2**2 + t**2)
i2z = (1/12)*m2*(t**2 + t**2)
Ms[2] = ca.diag(ca.DM([m2,m2,m2,i2x,i2y,i2z]))
Ms[3] = ca.diag(ca.DM([m2,m2,m2,i2x,i2y,i2z]))
Ms[4] = ca.diag(ca.DM([m1,m1,m1,i1,i1,i1])) 
Ms[5] = ca.diag(ca.DM([m1,m1,m1,i1,i1,i1])) 
m6 = 0.2
i6x = i6y = (1/12)*m6*((2*racket_r)**2 + (2*racket_r)**2)
i6z = i6x + i6y
Ms[6] = ca.diag(ca.DM([m6,m6,m6,i6x,i6y,i6z])) 

M = Js[1].T @ Ms[1] @ Js[1] + Js[2].T @ Ms[2] @ Js[2] + Js[3].T @ Ms[3] @ Js[3] + Js[4].T @ Ms[4] @ Js[4] + Js[5].T @ Ms[5] @ Js[5] + Js[6].T @ Ms[6] @ Js[6]

dM_dx = ca.jacobian(M, x)
dM_dx = [ca.reshape(dM_dx[:,i],6,6) for i in range(12)]
C = ca.MX.zeros(6, 6)
for i in range(6):
    for j in range(6):
        for k in range(6):
            C[i,j] = C[i,j] + (0.5*dM_dx[k][i,j] + 0.5*dM_dx[j][i,k] - 0.5*dM_dx[i][k,j])*q_dot[k]


gravity = ca.DM(9.8)
PE = gs[1][2,3]*Ms[1][0,0]*gravity + gs[2][2,3]*Ms[2][0,0]*gravity + gs[3][2,3]*Ms[3][0,0]*gravity + gs[4][2,3]*Ms[4][0,0]*gravity + gs[5][2,3]*Ms[5][0,0]*gravity + gs[6][2,3]*Ms[6][0,0]*gravity
N = ca.reshape(ca.jacobian(PE, x)[:,:6], (6,1))


q_ddot = ca.inv(M) @ (u - C@q_dot - N)
# q_ddot = ca.inv(M) @ (u - N)

x_dot = ca.vertcat(q_dot,q_ddot)

ee_x = ca.vertcat(decompose_transform(gs[6]), Js[6]@q_dot)

fk_func = ca.Function("fk_func", [x], gs)
dynamics_func = ca.Function("dynamics_func", [x, u], [x_dot])
ee_func = ca.Function("ee_func", [x], [ee_x])




# %%jit rk4
x = ca.SX.sym('x', params['nx'])
u = ca.SX.sym('u', params['nu']) if params['nu'] > 0 else []
dt = ca.SX.sym('dt')

x_rk4 = rk4(params,ball_flight_dynamics, x, u, dt)
rk4_jit = ca.Function('rk4_jit', [x, dt], [x_rk4], {'jit': False})
rk4_jit(np.zeros(12), 0.01) # compile

table_contact = check_table_contact(params, x)
check_table_contact_jit = ca.Function('check_table_contact_jit', [x], [table_contact], {'jit': False})
check_table_contact_jit(np.zeros(12)) # compile

# %%
# ball_opti

params['N'] = params['n1'] + params['n2'] + params['n3'] + params['n4'] + 1

nx = params['nx']
n1 = params['n1']
n2 = params['n2']
n3 = params['n3']
n4 = params['n4']
N = params['N']


# forward rollout for intial guess
dt = 0.01
ball_xs_rollout = np.zeros((nx, N))
ball_xs_rollout[:,0] = params['x0']
s = time.time()
for i in range(N-1):
    ball_xs_rollout[:,i+1] = rk4_jit(ball_xs_rollout[:,i], dt).toarray().reshape(params['nx'],).copy()
    if check_table_contact(params, ball_xs_rollout[:,i+1]):
        ball_xs_rollout[:,i+1] = table_contact_reset_map(params, ball_xs_rollout[:,i+1], None, ball_xs_rollout[:,i]).toarray().reshape(params['nx'],).copy()
print(f'Time for forward rollout => {time.time()-s}')

opti = ca.Opti()
Xs = opti.variable(nx, N)
hs = opti.variable(N-1)
x_r = opti.variable(nx)

# opti.set_initial(Xs, np.ones((nx, N)))
# opti.set_initial(hs, np.ones(N-1))
# opti.set_initial(x_r, np.ones(nx))

opti.set_initial(Xs, ball_xs_rollout)
opti.set_initial(hs, np.ones(N-1)*dt)

# opti.set_initial(Xs, sol_Xs)
# opti.set_initial(hs, sol_hs)


J = 0
J += ca.sumsqr(hs)
J += ca.sumsqr(Xs[:,N-1][0:2] - params['xg'][0:2])*10000
J += ca.sumsqr(x_r[6:])
J += ca.sumsqr(ca.sum1(hs[n1+n2:])-params['tg'])*10000
# J += ca.sumsqr(Xs[:,N-1][10] - params['xg'][10])*10000
opti.minimize(J)

opti.subject_to((transform_matrix(x_r[0], x_r[1], x_r[2], x_r[3], x_r[4], x_r[5])[:3,:3]@ca.DM([0,1,0]))[1] * x_r[1] < 0)

opti.subject_to((Xs[:,0]) - params['x0'] == 0)

for i in range(n1):
    opti.subject_to(Xs[:,i+1] - rk4(params, ball_flight_dynamics, Xs[:,i], None, hs[i]) == 0)
    # opti.subject_to(Xs[:,i+1] - rk4_jit(Xs[:,i], hs[i]) == 0)

opti.subject_to(Xs[:,n1][2]>params['ball_r'])
opti.subject_to(rk4(params, ball_flight_dynamics, Xs[:,n1], None, hs[n1])[2]<params['ball_r'])
opti.subject_to(Xs[:,n1+1] - table_contact_reset_map(params, rk4(params, ball_flight_dynamics, Xs[:,n1], None, hs[n1]), None, Xs[:,n1]) == 0)

for i in range(n1+1, n1+n2):
    opti.subject_to(Xs[:,i+1] - rk4(params, ball_flight_dynamics, Xs[:,i], None, hs[i]) == 0)
    # opti.subject_to(Xs[:,i+1] - rk4_jit(Xs[:,i], hs[i]) == 0)

opti.subject_to(Xs[:,n1+n2+1] - racket_contact_reset_map(params, rk4(params, ball_flight_dynamics, Xs[:,n1+n2], None, hs[n1+n2]), x_r, Xs[:,n1+n2]) == 0)


for i in range(n1+n2+1, n1+n2+n3):
    opti.subject_to(Xs[:,i+1] - rk4(params, ball_flight_dynamics, Xs[:,i], None, hs[i]) == 0)
    # opti.subject_to(Xs[:,i+1] - rk4_jit(Xs[:,i], hs[i]) == 0)

opti.subject_to(Xs[:,n1+n2+n3][2]>params['ball_r'])
opti.subject_to(rk4(params, ball_flight_dynamics, Xs[:,n1+n2+n3], None, 0.01)[2]<params['ball_r'])


opti.subject_to(0<=hs)
opti.subject_to(hs<=0.1)
opti.subject_to(hs[n1+n2]<=0.001)
opti.subject_to(hs[n1]<=0.001)
# opti.subject_to(ca.sum1(hs)==params['tg'])
# opti.subject_to(ca.sum1(hs[n1+n2:])==params['tg'])
opti.subject_to(ca.sumsqr(Xs[:,n1+n2+1][:3]-np.array([-1.6,0,0]))<=0.6**2)
opti.subject_to(x_r[:3]==Xs[:,n1+n2+1][:3])
# opti.subject_to(x_r[4]<=np.pi/2)
# opti.subject_to(x_r[4]>=-np.pi/2)

p_opts = {"expand":True,
          "jit": False,
          "verbose":False,}
s_opts = {"max_iter": 2000,
          "tol":1e-3,
          "print_level":5,}
opti.solver("ipopt",p_opts,s_opts)

try:
    sol = opti.solve()
except:
    print('optimum not found!')
    sol = opti.debug
sol_Xs = sol.value(Xs)
sol_hs = sol.value(hs)
# hs_m = np.insert(sol_hs, 0, 0)
hs_m = sol_hs
print(sol_hs.shape)
# print(sol.value(Xs[:,0]))
# print(sol.value(Xs[:,1]))

# %% 
# m_opti

# N = 100
x0 = np.array([0,0,0 , 0,0,0 , 0,0,0 ,0,0,0]).reshape(12,1) # joint angles
ee_hit = sol.value(x_r) # desired racket hitting pos,vel
# ee_hit[3:6] = ca.reshape(ca.atan2(ca.sin(ee_hit[3:6]), ca.cos(ee_hit[3:6])))
ee_hit[3:6] = np.arctan2(np.sin(ee_hit[3:6]), np.cos(ee_hit[3:6])) # wrapping roll, pitch, yaw angles between -pi,pi
# ee_hit[5] += np.pi 
xg = np.array([0,0,0, 0,0,0, 0,0,0, 0,0,0]) # end effector pos,vel

m_opti = ca.Opti()
Xs_m = m_opti.variable(18, N)
# m_opti.set_initial(Xs_m, np.random.rand(18,N))
m_opti.set_initial(Xs_m, np.zeros((18,N)))


J = ca.MX(0.0)
J += ca.sumsqr(Xs_m[:12,N-1] - xg) # goal position and vel at last timestep
# J += ca.sumsqr(fk_func(Xs_m[:12,n1+n2+1])[6] - transform_matrix(*ee_hit[:6]))
# J += ca.sumsqr(ee_func(Xs_m[:12,n1+n2+1])[6:] - ee_hit[6:])
J += ca.sumsqr(Xs_m[12:,:])/1e6 # restrict acceleration
m_opti.minimize(J)

# m_opti.subject_to(ca.sumsqr(fk_func(Xs_m[:12,n1+n2+1])[6] - transform_matrix(*ee_hit[:6]))<=1e-3)
m_opti.subject_to(ca.sumsqr(fk_func(Xs_m[:12,n1+n2])[6] - transform_matrix(*ee_hit[:6]))<=1e-8)
m_opti.subject_to(ca.sumsqr(ee_func(Xs_m[:12,n1+n2])[6:] - ee_hit[6:])<=1e-8)

m_opti.subject_to(ca.vec(Xs_m[:6,n1+n2])<np.pi)
m_opti.subject_to(ca.vec(Xs_m[:6,n1+n2])>-np.pi)

m_opti.subject_to((Xs_m[:12,0]) - x0 == 0) # initial condition
# m_opti.subject_to((ee_func(Xs_m[:12,n1+n2+1])) - ee_hit == 0)

# kinematic constraints
for i in range(N-1):
    dt = hs_m[i]
    m_opti.subject_to(Xs_m[:6,i+1] - Xs_m[:6,i] - Xs_m[6:12,i]*dt == 0)
    m_opti.subject_to(Xs_m[6:12,i+1] - Xs_m[6:12,i] - Xs_m[12:,i]*dt == 0)

p_opts = {"expand":False,
          "verbose":False,}
s_opts = {"max_iter": 1000,
          "tol":1e-3,
          #"hessian_approximation": "limited-memory",
          'print_level': 3,
          "constr_viol_tol": 1e-3,}

m_opti.solver("ipopt",p_opts,s_opts)

try:
    s = time.time()
    m_sol = m_opti.solve()
    print(f'OPTI STACK TIME  = {time.time()-s}')
except:
    print('optimum not found!')
    m_sol = m_opti.debug
sol_Xs_m = m_sol.value(Xs_m)

print("expected position: ",transform_matrix(ee_hit[0],ee_hit[1],ee_hit[2],ee_hit[3],ee_hit[4],ee_hit[5]))
print("actual position: ", fk_func(sol_Xs_m[:12,N-1])[6])

print("ee_func ", ee_func(sol_Xs_m[:12,n1+n2+1])[:6])
# print(sol_us)

print("expected velocity: ", ee_hit[6:])
print("actual velocity: ", ee_func(sol_Xs_m[:12,N-1])[6:])

# print(sol_Xs_m)

# %%

rend = Renderer(params)
rend.render_init(fk_func)
ts = np.cumsum(np.insert(sol_hs, 0, 0))
# rend.render_anim(ts, sol_x, traj=True)
x_r = sol.value(x_r)
x_r[0:3] = sol_Xs[:,n1+n2][0:3]
# rend.render_anim(ts, sol_Xs, racket1_xs=np.repeat(sol.value(x_r)[:,None], repeats=N, axis=1), traj=True)
anim2 = rend.render_anim(ts, sol_Xs, racket1_xs=np.repeat(sol.value(x_r)[:,None], repeats=N, axis=1), m1_xs = sol_Xs_m, fk_func=fk_func,traj=True)
rend.vis.set_property("/Animations/default/timeScale", 0.1)

# self, ts, ball_xs, racket1_xs=None, racket2_xs=None, m1_xs=None, fk_func=None,  traj=False
rend.vis.jupyter_cell()
# %%

rend = Renderer(params)
rend.render_init(fk_func)
rend.vis.open()
time.sleep(0.5)
s = None
for i in range(5):
    # params['x0'] = np.array([1.5,0,0.25, 
    #                          0,0,0,
    #                          np.random.uniform(-5,-4), np.random.uniform(-1,-0.5), np.random.uniform(2,3),
    #                          np.random.uniform(-200,200), np.random.uniform(-200,200), np.random.uniform(-200,200)])

    # params['N'] = params['n1'] + params['n2'] + params['n3'] + params['n4'] + 1

    # nx = params['nx']
    # n1 = params['n1']
    # n2 = params['n2']
    # n3 = params['n3']
    # n4 = params['n4']
    # N = params['N']

    # forward rollout for intial guess
    dt = 0.01
    ball_xs_rollout = np.zeros((nx, N))
    ball_xs_rollout[:,0] = params['x0']
    s = time.time()
    for i in range(N-1):
        ball_xs_rollout[:,i+1] = rk4_jit(ball_xs_rollout[:,i], dt).toarray().reshape(params['nx'],).copy()
        if check_table_contact(params, ball_xs_rollout[:,i+1]):
            ball_xs_rollout[:,i+1] = table_contact_reset_map(params, ball_xs_rollout[:,i+1], None, ball_xs_rollout[:,i]).toarray().reshape(params['nx'],).copy()
    print(f'Time for forward rollout => {time.time()-s}')

    opti = ca.Opti()
    Xs = opti.variable(nx, N)
    hs = opti.variable(N-1)
    x_r = opti.variable(nx)

    # opti.set_initial(Xs, np.ones((nx, N)))
    # opti.set_initial(hs, np.ones(N-1))
    # opti.set_initial(x_r, np.ones(nx))

    opti.set_initial(Xs, ball_xs_rollout)
    opti.set_initial(hs, np.ones(N-1)*dt)

    # opti.set_initial(Xs, sol_Xs)
    # opti.set_initial(hs, sol_hs)


    J = 0
    J += ca.sumsqr(hs)
    J += ca.sumsqr(Xs[:,N-1][0:2] - params['xg'][0:2])*10000
    J += ca.sumsqr(x_r[6:])
    J += ca.sumsqr(ca.sum1(hs[n1+n2:])-params['tg'])*10000
    opti.minimize(J)

    opti.subject_to((transform_matrix(x_r[0], x_r[1], x_r[2], x_r[3], x_r[4], x_r[5])[:3,:3]@ca.DM([0,1,0]))[1] * x_r[1] < 0)

    opti.subject_to((Xs[:,0]) - params['x0'] == 0)

    for i in range(n1):
        opti.subject_to(Xs[:,i+1] - rk4(params, ball_flight_dynamics, Xs[:,i], None, hs[i]) == 0)
        # opti.subject_to(Xs[:,i+1] - rk4_jit(Xs[:,i], hs[i]) == 0)

    opti.subject_to(Xs[:,n1][2]>params['ball_r'])
    opti.subject_to(rk4(params, ball_flight_dynamics, Xs[:,n1], None, hs[n1])[2]<params['ball_r'])
    opti.subject_to(Xs[:,n1+1] - table_contact_reset_map(params, rk4(params, ball_flight_dynamics, Xs[:,n1], None, hs[n1]), None, Xs[:,n1]) == 0)

    for i in range(n1+1, n1+n2):
        opti.subject_to(Xs[:,i+1] - rk4(params, ball_flight_dynamics, Xs[:,i], None, hs[i]) == 0)
        # opti.subject_to(Xs[:,i+1] - rk4_jit(Xs[:,i], hs[i]) == 0)

    opti.subject_to(Xs[:,n1+n2+1] - racket_contact_reset_map(params, rk4(params, ball_flight_dynamics, Xs[:,n1+n2], None, hs[n1+n2]), x_r, Xs[:,n1+n2]) == 0)


    for i in range(n1+n2+1, n1+n2+n3):
        opti.subject_to(Xs[:,i+1] - rk4(params, ball_flight_dynamics, Xs[:,i], None, hs[i]) == 0)
        # opti.subject_to(Xs[:,i+1] - rk4_jit(Xs[:,i], hs[i]) == 0)

    opti.subject_to(Xs[:,n1+n2+n3][2]>params['ball_r'])
    opti.subject_to(rk4(params, ball_flight_dynamics, Xs[:,n1+n2+n3], None, 0.01)[2]<params['ball_r'])


    opti.subject_to(0<=hs)
    opti.subject_to(hs<=0.1)
    opti.subject_to(hs[n1+n2]<=0.001)
    opti.subject_to(hs[n1]<=0.001)
    # opti.subject_to(ca.sum1(hs)==params['tg'])
    # opti.subject_to(ca.sum1(hs[n1+n2:])==params['tg'])
    opti.subject_to(ca.sumsqr(Xs[:,n1+n2+1][:3]-np.array([-1.6,0,0]))<=0.6**2)
    opti.subject_to(x_r[:3]==Xs[:,n1+n2+1][:3])
    # opti.subject_to(x_r[4]<=np.pi/2)
    # opti.subject_to(x_r[4]>=-np.pi/2)

    p_opts = {"expand":True,
            "jit": False,
            "verbose":False,}
    s_opts = {"max_iter": 1000,
            "tol":1e-3,
            "print_level":5,}
    opti.solver("ipopt",p_opts,s_opts)

    try:
        sol = opti.solve()
    except:
        print('optimum not found!')
        sol = opti.debug
    sol_Xs = sol.value(Xs)
    sol_hs = sol.value(hs)
    # hs_m = np.insert(sol_hs, 0, 0)
    hs_m = sol_hs
    print(sol_hs.shape)
    # print(sol.value(Xs[:,0]))
    # print(sol.value(Xs[:,1]))

    # m_opti

    # N = 100
    x0 = np.array([0,0,0 , 0,0,0 , 0,0,0 ,0,0,0]).reshape(12,1) # joint angles
    ee_hit = sol.value(x_r) # ball final pos,vel
    # ee_hit[3:6] = ca.reshape(ca.atan2(ca.sin(ee_hit[3:6]), ca.cos(ee_hit[3:6])))
    ee_hit[3:6] = np.arctan2(np.sin(ee_hit[3:6]), np.cos(ee_hit[3:6]))
    # ee_hit[5] += np.pi 
    xg = np.array([0,0,0, 0,0,0, 0,0,0, 0,0,0]) # end effector pos,vel

    m_opti = ca.Opti()
    Xs_m = m_opti.variable(18, N)
    # m_opti.set_initial(Xs_m, np.random.rand(18,N))
    m_opti.set_initial(Xs_m, np.zeros((18,N)))


    J = ca.MX(0.0)
    J += ca.sumsqr(Xs_m[:12,N-1] - xg)
    # J += ca.sumsqr(fk_func(Xs_m[:12,n1+n2+1])[6] - transform_matrix(*ee_hit[:6]))
    # J += ca.sumsqr(ee_func(Xs_m[:12,n1+n2+1])[6:] - ee_hit[6:])
    J += ca.sumsqr(Xs_m[12:,:])/1e6 # restrict acceleration
    m_opti.minimize(J)


    # m_opti.subject_to(ca.sumsqr(fk_func(Xs_m[:12,n1+n2+1])[6] - transform_matrix(*ee_hit[:6]))<=1e-3)
    m_opti.subject_to(ca.sumsqr(fk_func(Xs_m[:12,n1+n2])[6] - transform_matrix(*ee_hit[:6]))<=1e-6)
    m_opti.subject_to(ca.sumsqr(ee_func(Xs_m[:12,n1+n2])[6:] - ee_hit[6:])<=1e-6)

    m_opti.subject_to(ca.vec(Xs_m[:6,:])<np.pi)
    m_opti.subject_to(ca.vec(Xs_m[:6,:])>-np.pi)

    m_opti.subject_to((Xs_m[:12,0]) - x0 == 0)
    # m_opti.subject_to((ee_func(Xs_m[:12,n1+n2+1])) - ee_hit == 0)

    for i in range(N-1):
        dt = hs_m[i]
        m_opti.subject_to(Xs_m[:6,i+1] - Xs_m[:6,i] - Xs_m[6:12,i]*dt == 0)
        m_opti.subject_to(Xs_m[6:12,i+1] - Xs_m[6:12,i] - Xs_m[12:,i]*dt == 0)

    p_opts = {"expand":False,
            "verbose":False,}
    s_opts = {"max_iter": 1000,
            "tol":1e-3,
            #"hessian_approximation": "limited-memory",
            'print_level': 5,
            "constr_viol_tol": 1e-3,}

    m_opti.solver("ipopt",p_opts,s_opts)

    try:
        s = time.time()
        m_sol = m_opti.solve()
        print(f'OPTI STACK TIME  = {time.time()-s}')
    except:
        print('optimum not found!')
        m_sol = m_opti.debug
    sol_Xs_m = m_sol.value(Xs_m)

    if s == None:
        pass
    elif time.time()-s <= ts[-1]:
        time.sleep(0.5)
    
    slow_factor = 10
    ts = np.cumsum(np.insert(sol_hs, 0, 0))*2
    # ts = np.cumsum(np.insert(sol_hs, 0, 0))
    x_r = sol.value(x_r)
    x_r[0:3] = sol_Xs[:,n1+n2][0:3]
    s = time.time()
    rend.render_anim(ts, sol_Xs, racket1_xs=np.repeat(sol.value(x_r)[:,None], repeats=N, axis=1), m1_xs = sol_Xs_m, fk_func=fk_func,traj=True)
    time.sleep(1)




# %%
