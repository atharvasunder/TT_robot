# %%
import numpy as np
import scipy
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time

from tkApp import NonBlockingTkinterApp

# %%

class TTSim:

    def __init__(self, dt):
        self.dt = dt

        self.table_l = 2.74
        self.table_w = 1.525
        self.table_h = 0.012

        self.net_l = 0.002
        self.net_w = 1.83
        self.net_h = 0.1525

        self.ball_r = 0.02
        self.ball = Ball()

        self.traj_history = []

        self.pause = True



    def step(self):
        self.ball.step(dt=self.dt)
        # self.x[0:3] = np.array(init_pos)
        # self.x[3:6] = np.array(init_rot)
        # self.x[6:9] = np.array(init_lin_vel)
        # self.x[9:12] = np.array(init_ang_vel)
        if (np.abs(self.ball.x[0])<=self.table_l/2) and (np.abs(self.ball.x[1])<=self.table_w/2) and (self.ball.x[2]<=self.ball_r):

            x_out = np.copy(self.ball.x)
            mu = 0.102 # coeff of friction
            epsilon = 0.883 # coedd of restitution 
            tangential_vel = np.array([self.ball.x[6]-self.ball_r*self.ball.x[10], self.ball.x[7]-self.ball_r*self.ball.x[9], 0])
            alpha = mu*(1+epsilon)*np.abs(self.ball.x[8])/np.linalg.norm(tangential_vel)
            if alpha<=0.4:
            # alpha = mu*(1+epsilon)*self.ball.x[8]/np.linalg.norm(tangential_vel)
                A_v = np.diag([1-alpha, 1-alpha, -epsilon])
                B_v = np.zeros((3,3))
                B_v[0,1] = alpha*self.ball_r
                B_v[1,0] = -alpha*self.ball_r
                
                A_w = np.zeros((3,3))
                A_w[0,1] = -3*alpha/2/self.ball_r
                A_w[1,0] = 3*alpha/2/self.ball_r
                B_w = np.diag([1-3*alpha/2, 1-3*alpha/2, 1])

                x_out[0:3] = self.ball.last_x[0:3]
                x_out[3:6] = self.ball.x[3:6]
                x_out[6:9] = A_v@self.ball.x[6:9] + B_v@self.ball.x[9:12]
                x_out[9:12] = A_w@self.ball.x[6:9] + B_w@self.ball.x[9:12]
                self.ball.x = np.copy(x_out)
            else:
                A_v = np.diag([3/5, 3/5, -epsilon])
                B_v = np.zeros((3,3))
                B_v[0,1] = 2*self.ball_r/5
                B_v[1,0] = -2*self.ball_r/5
                
                A_w = np.zeros((3,3))
                A_w[0,1] = -3/5/self.ball_r
                A_w[1,0] = 3/5/self.ball_r
                B_w = np.diag([2/5, 2/5, 1])

                x_out[0:3] = self.ball.last_x[0:3]
                x_out[3:6] = self.ball.x[3:6]
                x_out[6:9] = A_v@self.ball.x[6:9] + B_v@self.ball.x[9:12]
                x_out[9:12] = A_w@self.ball.x[6:9] + B_w@self.ball.x[9:12]
                self.ball.x = np.copy(x_out)


    def render_init(self):
        self.vis = meshcat.Visualizer().open()
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



    def render(self, traj=True):

        rotation_matrix = tf.euler_matrix(*self.ball.x[3:6])
        translation_matrix = tf.translation_matrix(self.ball.x[0:3]) 
        axis_transform = translation_matrix @ rotation_matrix
        self.vis['ball'].set_transform(axis_transform) 
        # self.vis['ball'].set_transform(tf.translation_matrix(self.ball.x[0:3])) 

            
        if traj==True:
            self.traj_history.append(self.ball.x[0:3])
            self.vis['traj'].set_object(g.Line(g.PointsGeometry(np.array(self.traj_history).T)))           
        time.sleep(self.dt)
        # time.sleep(0.5)
        
        pass



class Ball:

    def __init__(self, init_pos=[0,0,0], init_rot=[0,0,0], init_lin_vel=[0,0,0], init_ang_vel=[0,0,0]):
        self.x = np.zeros((12))
        self.x[0:3] = np.array(init_pos)
        self.x[3:6] = np.array(init_rot)
        self.x[6:9] = np.array(init_lin_vel)
        self.x[9:12] = np.array(init_ang_vel)
        self.last_x = self.x
    
    def set_state(self, init_pos=[0,0,0], init_rot=[0,0,0], init_lin_vel=[0,0,0], init_ang_vel=[0,0,0]):
        self.x[0:3] = np.array(init_pos)
        self.x[3:6] = np.array(init_rot)
        self.x[6:9] = np.array(init_lin_vel)
        self.x[9:12] = np.array(init_ang_vel)



    # def calc_dx(self):
    #     self.ddx = np.zeros((12))
    #     self.ddx[0:3] = self.x[6:9]
    #     self.ddx[3:6] = self.x[9:12]
    #     self.ddx[6:9] = np.array([0, 0, self.g]).T - self.C_D*np.linalg.norm(self.x[6:9])*self.x[6:9] + self.C_L*np.cross(self.x[9:12], self.x[6:9])
    #     self.ddx[9:12] = np.zeros_like(self.ddx[9:12])
    #     return self.ddx

    @staticmethod 
    def calc_dx_scipy(t, y):
        # for scipy.integrate.solve_ivp, state has to be 1D
        g = -9.8
        C_D = 0.141
        C_L = 0.001
        dx = np.zeros((12))
        dx[0:3] = y[6:9]
        dx[3:6] = y[9:12]
        dx[6:9] = np.array([0, 0, g]).T - C_D*np.linalg.norm(y[6:9])*y[6:9] + C_L*np.cross(y[9:12], y[6:9])
        dx[9:12] = np.zeros_like(y[9:12])
        return dx   


    def step(self, dt=0.1):
        sol = scipy.integrate.solve_ivp(self.calc_dx_scipy, [0, dt], self.x, t_eval=[dt])
        self.last_x = self.x
        self.x = sol.y[:,-1]


def pause_button_callback(sim):
    sim.pause = True if sim.pause==False else False

def reset_button_callback(sim):
    # sim.ball.set_state(init_pos=[-1.5,0,0.25], init_lin_vel=[5, 0, 1], init_ang_vel=[1000,0,0])
    sim.traj_history = []
    sim.ball.set_state(init_pos=eval(sim.app.init_pos.get()), init_rot=eval(sim.app.init_rot.get()), init_lin_vel=eval(sim.app.init_lin_vel.get()), init_ang_vel=eval(sim.app.init_ang_vel.get()))
    sim.step()
    sim.render()

# %%
if __name__=="__main__":
# %%
    sim = TTSim(dt=0.001)
    sim.render_init()
    sim.ball.set_state(init_pos=[-1.5,0,0.25], init_lin_vel=[5, 0, 1], init_ang_vel=[1000,0,0])

    app = NonBlockingTkinterApp()
    sim.app = app
    app.bind_button('Pause/Play', pause_button_callback, sim, row=0, column=0)
    app.bind_button('Reset', reset_button_callback, sim, row=0, column=1)
    app.init_pos = app.bind_entry_with_label('init_pos', str([-1.5,0,0.25]), row=1)
    app.init_rot = app.bind_entry_with_label('init_rot', str([0,0,0]), row=2)
    app.init_lin_vel = app.bind_entry_with_label('init_lin_vel', str([5,0,1]), row=3)
    app.init_ang_vel = app.bind_entry_with_label('init_ang_vel', str([0,0,500]), row=4)
    # import ipdb; ipdb.set_trace()
    while True:
        app.update()
        if sim.pause==True:
            pass
        else:
            sim.step()
            sim.render()
    # print(ball.x)

    
# %%
