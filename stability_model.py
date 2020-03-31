"""
Simple model of standing postural stability, consisting of foot and body segments,
and two muscles that create moments about the ankles, tibialis anterior and soleus.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from musculoskeletal import HillTypeMuscle, get_velocity, force_length_tendon


def tibialis_length(theta):
    """
    :param theta: body angle (up from prone horizontal)  # TODO: CHANGE THETA TO DOWN FROM PRONE HORIZONTAL
    :return: tibialis anterior length
    """
    rotation = [[np.cos(theta + np.pi / 2), -np.sin(theta + np.pi / 2)], [np.sin(theta + np.pi / 2), np.cos(theta + np.pi / 2)]]
    origin = np.dot(rotation, [.3, -.03])
    insertion = [.06, -.03]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def gravity_moment(theta, foot_mass, d_COM):
    """
    :param theta: angle of body segment (up from prone)
    :param foot_mass: mass of body (in kg)
    :param d_COM: distance from ankle to center of mass of foot (in m)
    :return: moment about ankle due to force of gravity on foot
    """
    g = 9.81 # acceleration of gravity
    return foot_mass * g * d_COM * np.cos(theta)


def dynamics(x, t, tibialis):
    """
    :param x: state vector [ankle angle, angular velocity, horizontal coordinate of ankle joint,
                            horizontal velocity of ankle, horizontal acceleration of ankle,
                            vertical position of ankle, velocity of ankle in y direction,
                            acceleration of ankle in y direction, normalized length of TA muscle]
    :param t: time
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :return: derivative of state vector
    """
    i_ankle = 90  # kgm^2
    dta = .03  # m

    torque_ta = tibialis.f0M * force_length_tendon(tibialis.norm_tendon_length(tibialis_length(x[0]), x[3])) * dta

    x_dot = np.empty(x.shape)
    # TODO: Define length_of_foot, mass_of_foot, d_COM (centre_of_mass_distance), ls (stride length), sigma
    # TODO: Define Ms such that Ms / (sigma * sqrt(2 * pi)) = maximum vertical displacement during swing phase
    # TODO: Define activation of TA

    # modeling the average male
    body_mass = 90  # kg
    foot_length = 0.254  # metres (10 inches)
    foot_mass = 0.0137 * body_mass  # mass of foot (kg)  --> female is 1.29
    d_COM = 0.4415 * foot_length  # distance from ankle to centre of mass of foot (m)
    ls = 0.5 # TODO: find stride length
    sigma = 0.05
    max_vertiical_disp = 0.1 #TODO: max vertical displacement disp
    ms = 0.1 * sigma * np.sqrt(2 * np.pi)
    a_ta = 0.1

    x_dot[0] = x[1]
    x_dot[1] = ((torque_ta - gravity_moment(x[0], foot_mass, foot_length))
                + foot_mass * d_COM * (x[4] * np.sin(x[0] - x[7] * np.cos(x[0])) / i_ankle))
    x_dot[2] = x[3]
    x_dot[3] = x[4]
    x_dot[4] = (- ls / ((sigma ** 3) * np.sqrt(2 * np.pi))) * np.exp(- ((t - 0.7) ** 2) / (2 * (sigma ** 2))) + (ls / ((sigma ** 5) * np.sqrt(2 * np.pi))) * (t - 0.7) **2 * np.exp(- ((t - 0.7) ** 2) / (2 * (sigma ** 2)))
    # x_dot[4] = (-1/sigma ** 2) * x[3] - (t/sigma ** 2) * x[4]
    x_dot[5] = x[6]
    x_dot[6] = x[7]
    x_dot[7] = (ms / ((sigma ** 5) * np.sqrt(2 * np.pi))) * np.exp(- ((t - 0.7) ** 2) / (2 * (sigma ** 2))) * (3 * (t - 0.7) - (t - 0.7) ** 3 / sigma ** 2)
    # x_dot[7] = (3/ sigma ** 2) * x[6] + (t ** 2/ sigma ** 4) * x[6]
    x_dot[8] = get_velocity(a_ta, x[8], tibialis.norm_tendon_length(tibialis_length(x[0]), x[8]))

    y = x[5] - foot_length * np.sin(x[0])

    return x_dot

def get_toe_clearance(x, foot_length):
    return x[5, :] - foot_length * np.sin(x) #TODO: need to consider the heights of the ankle above the foot and the initia angle of the foot naturally

def simulate(control, T):
    """
    Runs a simulation of the model and plots results.
    :param control: True if balance should be controlled
    :param T: total time to simulate, in seconds (range)
    """
    rest_length_tibialis = tibialis_length(np.pi/2)

    tibialis = HillTypeMuscle(2000, .6*rest_length_tibialis, .4*rest_length_tibialis)

    def f(t, x):
        return dynamics(x, t, tibialis)

    sol = solve_ivp(f, T, [0, 0, 0, 0, 0, 0, 0, 0, 1], rtol=1e-5, atol=1e-8)
    time = sol.t
    theta = sol.y[0,:]
    tibialis_norm_length_muscle = sol.y[8,:]

    tibialis_moment_arm = .03
    tibialis_moment = []
    for th, lt in zip(theta, tibialis_norm_length_muscle):
        tibialis_moment.append(-tibialis_moment_arm * tibialis.get_force(tibialis_length(th), lt))

    body_mass = 90  # kg
    foot_length = 0.254  # metres (10 inches)
    foot_mass = 0.0137 * body_mass  # mass of foot (kg)  --> female is 1.29
    d_COM = 0.4415 * foot_length  # distance from ankle to centre of mass of foot (m)

    plt.figure()
    plt.subplot(4, 3, 1)
    plt.plot(time, sol.y[0, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Ankle Angle (rad)")
    plt.subplot(4, 3, 2)
    plt.plot(time, sol.y[1, :])
    plt.ylabel("Angular Velocity (rad/s)")
    plt.subplot(4, 3, 3)
    plt.plot(time, sol.y[2, :])
    plt.ylabel("Horizontal Position of Ankle (m)")
    plt.subplot(4, 3, 4)
    plt.plot(time, sol.y[3, :])
    plt.ylabel("Horizontal Velocity of Ankle (m/s)")
    plt.subplot(4, 3, 5)
    plt.plot(time, sol.y[4, :])
    plt.ylabel("Horizontal Acceleration of Ankle (m/s^2)")
    plt.subplot(4, 3, 6)
    plt.plot(time, sol.y[5, :])
    plt.ylabel("Vertical Position of Ankle (m)")
    plt.subplot(4, 3, 7)
    plt.plot(time, sol.y[6, :])
    plt.ylabel("Vertical Velocity of Ankle (m/s)")
    plt.subplot(4, 3, 8)
    plt.plot(time, sol.y[7, :])
    plt.ylabel("Vertical acceleration of ankle (m/s^2)")
    plt.subplot(4, 3, 9)
    plt.plot(time, sol.y[8, :])
    plt.ylabel("Normalized Length of TA")
    plt.tight_layout()
    plt.show()

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(time, sol.y[0,:])
    # plt.ylabel('Ankle angle (rad)')
    # plt.subplot(2,1,2)
    # plt.plot(time, tibialis_moment, 'g')
    # plt.plot(time, gravity_moment(sol.y[0,:], foot_mass, d_COM), 'k')
    # plt.legend(('tibialis', 'gravity'))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Torques (Nm)')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(time, sol.y[2, :])
    # plt.ylabel('Position')
    # plt.subplot(3, 1, 2)
    # plt.plot(time, sol.y[3, :], 'g')
    # plt.ylabel('Velocity')
    # plt.subplot(3, 1, 3)
    # plt.plot(time, sol.y[4, :], 'k')
    # plt.ylabel('Acceleration')
    # plt.xlabel('Time (s)')
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(3, 1, 1)
    # plt.plot(time, sol.y[5, :])
    # plt.ylabel('Position Y')
    # plt.subplot(3, 1, 2)
    # plt.plot(time, sol.y[6, :], 'g')
    # plt.ylabel('Velocity Y')
    # plt.subplot(3, 1, 3)
    # plt.plot(time, sol.y[7, :], 'k')
    # plt.ylabel('Acceleration Y')
    # plt.xlabel('Time (s)')
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    # simulate(False, (0.5, 0.9))
    simulate(False, (0, 1))