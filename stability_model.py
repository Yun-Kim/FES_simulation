"""
Simple model of standing postural stability, consisting of foot and body segments,
and two muscles that create moments about the ankles, tibialis anterior and soleus.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapz
from scipy.misc import derivative
from musculoskeletal import HillTypeMuscle
from musculoskeletal import get_shank_angle, get_shank_angle_comp
from seaborn import lineplot

torque = []
time_try = []
a_ta_array = []


def tibialis_length(theta):
    """
    :param theta: body angle (up from prone horizontal)  # TODO: CHANGE THETA TO DOWN FROM PRONE HORIZONTAL
    :return: tibialis anterior length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
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


def accel_x(t):
    ls = 0.40
    toe_off = 0.15
    h_max = 0.06
    sigma = 0.05
    T = np.sqrt(-2 * sigma ** 2 * np.log((toe_off - 0.1) / h_max))
    return -ls/(sigma**3 * np.sqrt(2*np.pi)) * t * np.exp(-(t - T)**2/(2*sigma**2))


def accel_y(t):
    toe_off = 0.15
    h_max = 0.06
    sigma = 0.05
    T = np.sqrt(-2 * sigma ** 2 * np.log((toe_off - 0.1) / h_max))
    return -h_max/sigma**2 * np.exp(-(t - T)**2/(2*sigma**2)) + h_max*t**2/sigma**4 * np.exp(-(t - T)**2/(2*sigma**2))


def pos_y(t):
    toe_off = 0.15
    h_max = 0.06
    sigma = 0.05
    T = np.sqrt(-2*sigma**2 * np.log((toe_off - 0.1)/h_max))
    return h_max * np.exp(-(t - T)**2/(2*sigma**2)) + 0.1


def get_force_length(alpha_s, theta):
    d_ta = 0.037  # Moment arm of TA w.r.t ankle (m)
    l_t = 0.223  # Constant tendon length (m)
    l_mt0 = 0.321  # Muscle tendon length at rest (m)
    W = 0.56  # Shape parameter of f_fl
    l_ce_optimal = 0.082  # From literature (m)

    l_ce = l_mt0 + d_ta * (alpha_s - theta) - l_t
    return np.exp(-((l_ce - l_ce_optimal) / (W * l_ce_optimal)) ** 2)


def get_force_velocity(alpha_s_dot, theta_dot):
    d_ta = 0.037  # Moment arm of TA w.r.t ankle (m)
    v_max = - 0.9  # Maximum contraction velocity (m/sec)
    a_v = 1.33  # Force-velocity parameter 1
    f_v1 = 0.18  # Force-velocity parameter 2
    f_v2 = 0.023  # Force-velocity parameter 3

    velocity_ce = d_ta * (alpha_s_dot - theta_dot)

    if velocity_ce < 0:  # Contraction
        return (1 - velocity_ce/v_max) / (1 + velocity_ce/(v_max * f_v1))
    else:  # Extension or isometric
        return (1 + (a_v * velocity_ce) / f_v2) / (1 + velocity_ce / f_v2)


def get_toe_clearance(x, foot_length, t):
    """
    :param x: state vector [ankle angle, angular velocity]
    :param foot_length: length of foot (in m)
    :param t: time (np.linspace)
    :return: np.array of toe clearance values from simulation. Negative values are grounded to zero (meaning drag foot)
    """
    toe_clearance = np.zeros(t.shape)
    for i in range(len(t)):
        clearance = pos_y(t[i]) + foot_length * np.sin(x[0][i]) + 0.06
        toe_clearance[i] = clearance if clearance >= 0 else 0
    return toe_clearance


def dynamics(x, t, tibialis, control, activation=-1.0):
    """
    :param activation: optional float (between 0 and 1) for testing constant activations, -1 otherwise
    :param x: state vector [ankle angle, angular velocity]
    :param t: time
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :return: derivative of state vector
    """
    def saturate(val):
        if val > 1:
            return 1
        if val < 0:
            return 0
        return val

    x_dot = np.empty(x.shape)
    model_shank_angle, _, _ = get_shank_angle()

    if control:
        if x[0] < 0:
            a_ta = saturate(-0.01 * x[0] + 0.1)
        else:
            a_ta = 0
    else:
        a_ta = 0.15

    if activation != -1:
        a_ta = activation

    d_ta = 0.037  # moment arm of TA wrt the ankle (m)
    foot_length = 0.2  # (m)
    foot_mass = 1.0275  # mass of foot (kg)
    # d_COM = 0.1145  # distance from ankle to centre of mass of foot (m)
    i_foot = 0.0197  # Inertia of the foot around ankle (kgm^2)

    alpha_s = model_shank_angle.eval(t) * np.pi/180
    alpha_s_dot = derivative(model_shank_angle.eval, t, 0.01) * np.pi/180

    force_ta = a_ta * tibialis.f0M * get_force_length(alpha_s, x[0]) * get_force_velocity(alpha_s_dot, x[1])
    torque_ta = force_ta * d_ta
    torque.append(torque_ta)
    time_try.append(t)
    a_ta_array.append(a_ta**2)

    # Ignore T_acc and T_ela for simplicity for now
    # a1 = 2.1
    # a2 = -0.08
    # a3 = -7.97
    # a4 = 0.19
    # a5 = -1.79
    # T_acc = foot_mass * d_COM * (accel_x(t) * np.sin(x[0]) - accel_y(t) * np.cos(x[0]))
    # T_ela = np.exp(a1 + a2*x[0]) - np.exp(a3 + a4*x[0]) + a5
    # B = 0.82 * (alpha_s_dot - alpha_s)
    x_dot[0] = x[1]
    x_dot[1] = (torque_ta - gravity_moment(x[0], foot_mass, foot_length) ) / i_foot

    return x_dot


def simulate(control, T, test=-1.0):
    """
    Runs a simulation of the model and plots results.
    :param test: optional float activation value (between 0 and 1)
    :param control: True if balance should be controlled
    :param T: total time to simulate, in seconds (range)
    """
    rest_length_tibialis = tibialis_length(np.pi/2)
    tibialis = HillTypeMuscle(600, .6*rest_length_tibialis, .4*rest_length_tibialis)

    def f(t, x):
        if test != -1:
            return dynamics(x, t, tibialis, False, activation=test)
        return dynamics(x, t, tibialis, control)

    theta = -1.3  # Rad
    sol = solve_ivp(f, T, [theta, 0], rtol=1e-5, atol=1e-8)

    return sol


if __name__ == '__main__':
    total_swing_phase_duration = 0.3306  # Corresponds to measured swing time (from literature)
    sol = simulate(True, (0, total_swing_phase_duration))
    time = sol.t
    foot_length = 0.2  # (m)
    toe_clearance = get_toe_clearance(sol.y, foot_length, time)

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(time, sol.y[0, :])
    plt.ylabel("Ankle Angle (rad)")
    plt.subplot(2, 2, 2)
    plt.plot(time, sol.y[1, :])
    plt.ylabel("Angular Velocity (rad/s)")
    plt.subplot(2, 2, 3)
    plt.plot(time, pos_y(time))
    plt.ylabel("Position Y (m)")
    plt.subplot(2, 2, 4)
    plt.plot(time, toe_clearance)
    plt.ylabel("Toe Clearance (m)")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    model_comp, _, _ = get_shank_angle_comp()
    data = np.genfromtxt('Foot_Angle_Ref.csv',
                         delimiter=',')  # Data from WebPlotDigitizer extracted into data.csv
    angle_ref = data[:, 1]
    t_ref = data[:, 0]

    # Toe clearance plot
    plt.figure()
    plt.plot(time, toe_clearance)
    plt.ylabel('Distance (m)')
    plt.xlabel('Time (s)')
    plt.show()

    # Activation Plot
    plt.figure()
    ax = plt.gca()
    ax.set_ylim([0, 1.1])
    lineplot(time_try, a_ta_array)
    plt.ylabel('Activation (%)')
    plt.xlabel('Time (s)')
    plt.show()

    print(trapz(a_ta_array, time_try))
    #
    # # Torque plot
    # plt.figure()
    # lineplot(np.array(time_try).T, np.reshape(np.array(torque), np.array(time_try).shape))
    # plt.ylabel("TA Moment (Nm)")
    # plt.xlabel("Time (s)")
    # plt.show()
    #
    # # Angle Angle Plot
    # plt.figure()
    # plt.plot(time, sol.y[0] * 180 / np.pi, label='Computed')
    # plt.plot(time, model_comp.eval(time), label='Ref Model Computed')
    # plt.scatter(t_ref, angle_ref, label='Ref Experimental Data')
    # plt.legend()
    # plt.ylabel('Ankle Angle (degrees)')
    # plt.xlabel('Time (s)')
    # plt.show()
    #
    # # Ankle Angular Velocity
    # plt.figure("Ankle Angular Velocity Comparison")
    # plt.plot(time, sol.y[1] * 180 / np.pi, label='Computed')
    # plt.plot(time, derivative(model_comp.eval, time, 0.001), label='Ref Model Computed')
    # plt.legend()
    # plt.ylabel('Ankle Angular Velocity (degrees/s)')
    # plt.xlabel('Time (s)')
    # plt.show()
    #
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(time, pos_y(time))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Pos Y (m)')
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(time, accel_y(time))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Accel Y')
    #
    # plt.subplot(2, 2, 3)
    # plt.plot(time, accel_x(time))
    # plt.xlabel('Time (s)')
    # plt.ylabel('Accel X')
    # plt.tight_layout()
    # plt.show()

    # Plot ankle angles as a function of time with different const activations
    # activation_0 = simulate(False, (0, total_swing_phase_duration), test=0.0)
    # activation_1 = simulate(False, (0, total_swing_phase_duration), test=0.1)
    # activation_2 = simulate(False, (0, total_swing_phase_duration), test=0.2)
    # activation_3 = simulate(False, (0, total_swing_phase_duration), test=0.3)
    # activation_4 = simulate(False, (0, total_swing_phase_duration), test=0.5)
    # activation_5 = simulate(False, (0, total_swing_phase_duration), test=0.9)
    # activation_6 = simulate(False, (0, total_swing_phase_duration), test=1.0)
    #
    # plt.figure()
    # plt.plot(activation_0.t, activation_0.y[0], label="a_ta=0")
    # plt.plot(activation_1.t, activation_1.y[0], label="a_ta=0.1")
    # plt.plot(activation_2.t, activation_2.y[0], label="a_ta=0.2")
    # plt.plot(activation_3.t, activation_3.y[0], label="a_ta=0.3")
    # plt.plot(activation_4.t, activation_4.y[0], label="a_ta=0.5")
    # plt.plot(activation_5.t, activation_5.y[0], label="a_ta=0.9")
    # plt.plot(activation_6.t, activation_6.y[0], label="a_ta=1")
    # plt.legend()
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ankle Angle (rad)")
    # plt.show()


