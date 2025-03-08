"""
## =========================================================================== ## 
MIT License
Copyright (c) 2020 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: lagrangian_dynamics_example.py
## =========================================================================== ## 
"""

# System (Default Lib.)
import sys
# Numpy (Array computing Lib.) [pip3 install numpy]
import numpy as np
# Mtaplotlib (Visualization Lib.) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# Integrate a system of ordinary differential equations (ODE) [pip3 install scipy]
from scipy.integrate import solve_ivp
# For animation manipulator time
import time

class Dynamics_Ctrl():
    def __init__(self, L, m, time, dt): # If you want a flexible number of inputs, use *args and **kwargs
        # << PUBLIC >> #
        # Arm Length [m]
        self.L  = [L[0], L[1]] 
        # Arm Length (1/2) - Center of Gravity [m]
        self.lg = [L[0]/2, L[1]/2]
        # Mass [kg]
        self.m  = [m[0], m[1]]
        # Moment of Inertia [kg.m^2]
        self.I  = [(1/3)*(m[0])*(L[0]**2), (1/3)*(m[1])*(L[1]**2)]
        # Gravitational acceleration [m/s^2]
        self.g  = 9.81
        # Initial Time Parameters (Calculation)
        self.t = (0, time)
    
    def forward_kinematics(self, theta_1, theta_2):
        """
        Forward Kinematics for a 2-DOF manipulator.
        Args:
            theta_1 (float): Array of joint angles for the first joint (in radians).
            theta_2 (float): Array of jopint angles for the second joint (in radians).
        
        Returns:
            x (float): x-coordinates of the end effector.
            y (float): y-coordinates of the end effector.
        """
        L1, L2 = self.L

        # Link 1 position and orientation
        x1 = L1 * np.cos(theta_1)  # Link 1 end x
        y1 = L1 * np.sin(theta_1)   # Link 1 end y

        # end effector position and orientation
        xf = L1 * np.cos(theta_1) + L2 * np.cos(theta_1 + theta_2)     
        yf = L1 * np.sin(theta_1) + L2 * np.sin(theta_1 + theta_2)
        
        return x1 , y1 , xf , yf
    
    def inverse_kinematics(self, x, y):
        """
        Inverse Kinematics for a 2-DOF manipulator.
        Args:
            x (float): x-coordinate of the end effector.
            y (float): y-coordinate of the end effector.
        
        Returns:
            theta_1 (float): Joint angle for the first joint (in radians).
            theta_2 (float): Joint angle for the second joint (in radians).
        """
        L1, L2 = self.L
        r = np.sqrt(x**2 + y**2)  # Distance from the origin to the end-effector
        
        # Check if the position is reachable
        if r > L1 + L2:
            raise ValueError("Position is out of reach!")
        
        # Compute theta_2 using the law of cosines
        cos_theta_2 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
        theta_2 = np.arccos(np.clip(cos_theta_2, -1.0, 1.0))  # Ensure the value is in [-1, 1]

        # Compute theta_1
        theta_1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta_2), L1 + L2 * np.cos(theta_2))
        
        return theta_1, theta_2
    
    def differential_kinematics(self, theta_1, theta_2, dtheta_1, dtheta_2):
        """
        Differential Kinematics for a 2-DOF manipulator.
        Args:
            theta_1 (float): Joint angle for the first joint (in radians).
            theta_2 (float): Joint angle for the second joint (in radians).
            dtheta_1 (float): Joint velocity for the first joint (in radians/sec).
            dtheta_2 (float): Joint velocity for the second joint (in radians/sec).
        
        Returns:
            dx (float): Linear velocity in the x direction.
            dy (float): Linear velocity in the y direction.
        """
        L1, L2 = self.L
        
        # Compute Jacobian matrix components
        J11 = -L1 * np.sin(theta_1) - L2 * np.sin(theta_1 + theta_2)
        J12 = -L2 * np.sin(theta_1 + theta_2)
        J21 = L1 * np.cos(theta_1) + L2 * np.cos(theta_1 + theta_2)
        J22 = L2 * np.cos(theta_1 + theta_2)
        
        # Jacobian matrix
        J = np.array([[J11, J12], [J21, J22]])

        # Joint velocity vector
        joint_velocities = np.array([dtheta_1, dtheta_2])

        # End effector velocities
        end_effector_velocity = np.dot(J, joint_velocities)
        dx, dy = end_effector_velocity

        return dx, dy
    
    def inverse_differential_kinematics(self, theta_1, theta_2, dx, dy):
        """
        Inverse Differential Kinematics for a 2-DOF manipulator.
        Args:
            theta_1 (float): Joint angle for the first joint (in radians).
            theta_2 (float): Joint angle for the second joint (in radians).
            dx (float): Desired linear velocity in the x direction.
            dy (float): Desired linear velocity in the y direction.
        
        Returns:
            dtheta_1 (float): Joint velocity for the first joint (in radians/sec).
            dtheta_2 (float): Joint velocity for the second joint (in radians/sec).
        """
        L1, L2 = self.L
        
        # Compute Jacobian matrix components
        J11 = -L1 * np.sin(theta_1) - L2 * np.sin(theta_1 + theta_2)
        J12 = -L2 * np.sin(theta_1 + theta_2)
        J21 = L1 * np.cos(theta_1) + L2 * np.cos(theta_1 + theta_2)
        J22 = L2 * np.cos(theta_1 + theta_2)
        
        # Jacobian matrix
        J = np.array([[J11, J12], [J21, J22]])

        # Desired end effector velocities
        end_effector_velocity = np.array([dx, dy])

        # Compute the joint velocities using the pseudo-inverse of the Jacobian
        # If the Jacobian is invertible, this gives the correct joint velocities
        J_pseudo_inverse = np.linalg.pinv(J)
        joint_velocities = np.dot(J_pseudo_inverse, end_effector_velocity)

        dtheta_1, dtheta_2 = joint_velocities

        return dtheta_1, dtheta_2

    def lagrangian_dynamics(self, t, q):
        """
        Description:
            For many applications with fixed-based robots we need to find a multi-body dynamics formulated as:

            M(\theta)\ddot\theta + b(\theta, \dot\theta) + g(\theta) = \tau

            M(\theta)                      -> Generalized mass matrix (orthogonal).
            \theta,\dot\theta,\ddot\theta  -> Generalized position, velocity and acceleration vectors.
            b(\theta, \dot\theta)          -> Coriolis and centrifugal terms.
            g(\theta)                      -> Gravitational terms.
            \tau                           -> External generalized forces.

            Euler-Lagrange equation:

            L = T - U

            T -> Kinetic Energy (Translation + Rotation Part): (1/2) * m * v^2 + (1/2) * I * \omega^2 -> with moment of inertia 
            U -> Potential Energy: m * g * h

        Args:
            (1) q[Float Array]: Instantaneous position of the Robot (2 Joints) -> Theta_{1, 2} and 1_Derivation Theta_{1,2}
            (2) time [Float]: time.
            
        Returns:
            (1) parameter{1}, parameter{3} [Float]: 1_Derivation Theta_{1,2}
            (2) parameter{2}, parameter{4} [Float]: 2_Derivation Theta_{1,2}
        """

        theta_1, dtheta_1, theta_2, dtheta_2 = q

        # Generalized mass matrix M(theta)
        M_Mat = np.array([
            [self.I[0] + self.I[1] + self.m[0] * (self.lg[0]**2) + self.m[1] * ((self.L[0]**2) + (self.lg[1]**2) + 2 * self.L[0] * self.lg[1] * np.cos(theta_2)), 
            self.I[1] + self.m[1] * ((self.lg[1]**2) + self.L[0] * self.lg[1] * np.cos(theta_2))], 
            [self.I[1] + self.m[1] * ((self.lg[1]**2) + self.L[0] * self.lg[1] * np.cos(theta_2)), 
            self.I[1] + self.m[1] * (self.lg[1]**2)]
        ])

        # Coriolis and centrifugal terms b(theta, theta_dot)
        b_Mat = np.array([
            [-self.m[1] * self.L[0] * self.lg[1] * dtheta_2 * (2 * dtheta_1 + dtheta_2) * np.sin(theta_2)], 
            [self.m[1] * self.L[0] * self.lg[1] * (dtheta_1**2) * np.sin(theta_2)]
        ])

        # Gravitational terms g(theta)
        g_Mat = np.array([
            [self.m[0] * self.g * self.lg[0] * np.cos(theta_1) + self.m[1] * self.g * (self.L[0] * np.cos(theta_1) + self.lg[1] * np.cos(theta_1 + theta_2))], 
            [self.m[1] * self.g * self.lg[1] * np.cos(theta_1 + theta_2)]
        ])

        # External torques τ
        tau_Mat = np.array([[0.0], [0.0]])

        # Solve for joint accelerations: M(theta) * q̈ = -b(theta, theta_dot) - g(theta) + τ
        ddtheta = np.linalg.solve(M_Mat, -b_Mat - g_Mat + tau_Mat)

        return np.array([dtheta_1, ddtheta[0, 0], dtheta_2, ddtheta[1, 0]])

    def display_result(self, initial):
        """
        Description:
            Function for calculating and displaying the results of Lagrangian Dynamics Calculation.
            
        Args:
            (1) q[Float Array]: qposition of the Robot (2 Joints) -> Theta_{1, 2} and 1_Derivation Theta_{1,2}
        """

        # print(type(initial))

        calc_r = solve_ivp(self.lagrangian_dynamics, self.t, initial, method='RK45',dense_output=True)
        print(calc_r)
        # print(np.shape(calc_r.y[1,:]))

        # Axes and Label initialization.
        ax1 = [0, 0, 0, 0]
        y_label1 = [r'$\theta_1$', r'$\dot\theta_1$', r'$\theta_2$', r'$\dot\theta_2$']

        # Display (Plot) variables.
        fig1, ((ax1[0], ax1[1]), (ax1[2], ax1[3])) = plt.subplots(2, 2)

        fig1.suptitle('Manipulator trajectory', fontweight ='normal')

        for i in range(len(ax1)):
            ax1[i].plot(calc_r.t, calc_r.y[i, :])
            ax1[i].grid()
            ax1[i].set_xlabel(r'time [s]', fontsize=20)
            ax1[i].set_ylabel(y_label1[i], fontsize=20)

        # display plot
        plt.show()

    def display_animation(self, initial):
        """
        Description:
            Function for animation the motion of the maniplator
            
        Args:
            (1) q[Float Array]: qposition of the Robot (2 Joints) -> Theta_{1, 2} and 1_Derivation Theta_{1,2}
        """

        fig, ax = plt.subplots(1, 1)

        # solve dynamics of manipulator
        calc_r = solve_ivp(self.lagrangian_dynamics, self.t, initial, method='RK45',dense_output=True)
        print(calc_r)
        # print(np.shape(calc_r.y[1,:]))

        # Real-time loop
        t_start = time.time()  # Start time
        t_current = 0
        animation_speed = 1

        # plot initialization 
        ax.set_xlim(-sum(self.L), sum(self.L))
        ax.set_ylim(-sum(self.L), sum(self.L))
        ax.set_xlabel('X position (m)')
        ax.set_ylabel('Y position (m)')
        ax.set_title('2-DOF Manipulator Animation')

        # Create the lines for the links
        link1, = ax.plot([], [], 'ro-', lw=3, markersize=8)
        link2, = ax.plot([], [], 'bo-', lw=3, markersize=8)
        # time_text = self.__ax2.text(-1.5, 1.5, '', fontsize=12)

        while t_current < self.t[1]:

            t_current = animation_speed*(time.time() - t_start)  # Get real-time elapsed time
            if t_current > self.t[1]:  
                break  # Stop when we reach the simulation end time
        
            # Evaluate solution at current time (MATLAB's `deval` equivalent)
            q_real_time = calc_r.sol(t_current)
            theta_1, _, theta_2, _ = q_real_time  # Extract joint angles at that instant of time

            x1 , y1 , xf , yf = self.forward_kinematics(theta_1, theta_2) # link positions at that instant

            # Update plot
            link1.set_data([0, x1], [0, y1])
            link2.set_data([x1, xf], [y1, yf])
            # time_text.set_text(f'Time: {t_current:.2f} s')

            plt.pause(0.01) # important for realistic updates

        # Set additional parameters for successful display of the robot environment
        plt.show()

def main():
    # Initialization of the Class (Control Dynamics - Lagrangian)
    # Input:
    #   (1) Length of Arms (Link 1, Link2) [Float Array]
    #   (2) Mass
    #   (3) Time [INT]
    #   (4) Derivation of the Time [Float]
    # Example:
    #   x = Dynamics_Ctrl([1.0, 1.0], [1.25, 2.0], 10, 0.1)

    lD_c = Dynamics_Ctrl([0.3, 0.3], [1.0, 1.0], 10, 0.01)
 
    # Initial position of the Robot (2 Joints) -> Theta_{1, 2} and 1_Derivation Theta_{1,2}
    initial_p = np.array([np.pi/3, 0.0, 0.0, 0.0])

    # Display the result of the calculation:
    # The figure with the resulting 1_Derivation Theta_{1,2}, 2_Derivation Theta_{1,2}
    lD_c.display_animation(initial_p)

if __name__ == '__main__':
    # sys.exit(main())
    main()