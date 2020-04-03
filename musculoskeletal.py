import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sklearn.linear_model import Ridge
from scipy.special import expit
from scipy.integrate import solve_ivp
import scipy


class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length. These are given as constructor arguments.
    """

    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        """
        :param f0M: maximum isometric force
        :param resting_length_muscle: actual length (m) of muscle (CE) that corresponds to
            normalized length of 1
        :param resting_length_tendon: actual length of tendon (m) that corresponds to
            normalized length of 1
        """
        self.f0M = f0M
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon
            complex (typically found from joint angles and musculoskeletal geometry)
        :param normalized_muscle_length: normalized length of the contractile element
            (the state variable of the muscle model)
        :return: normalized length of the tendon
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon

    def get_force(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """
        return self.f0M * force_length_tendon(self.norm_tendon_length(total_length, norm_muscle_length))


def get_velocity(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """
    beta = 0.1 # damping coefficient (see damped model in Millard et al.)
    # alpha assumed to be 0 --> cos term eliminated

    def func(vm):
        return a * force_length_muscle(lm) * force_velocity_muscle(vm) + force_length_parallel(lm) + beta * vm - force_length_tendon(lt)
    return fsolve(func, 0)[0]


def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """
    slack_length = 1
    if isinstance(lt, np.ndarray):
        result = np.empty(lt.shape)
        for i, length in enumerate(lt):
            if length < slack_length:
                result[i] = 0
            else:
                result[i] = 10 * (length - slack_length) + 240 * (length - slack_length) ** 2
    else:
        if lt < slack_length:
            result = 0
        else:
            result = 10 * (lt - slack_length) + 240 * (lt - slack_length)**2
    return result


def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """
    slack_length = 1
    if isinstance(lm, np.ndarray):
        result = np.empty(lm.shape)
        for i, length in enumerate(lm):
            if length < slack_length:
                result[i] = 0
            else:
                result[i] = (3 * (length - slack_length) ** 2) / (0.6 + length - 1)
    else:
        if lm < slack_length:
            result = 0
        else:
            result = (3 * (lm - slack_length) ** 2) / (0.6 + lm - 1)
    return result


def plot_curves():
    """
    Plot force-length, force-velocity, SE, and PE curves.
    """
    lm = np.arange(0, 1.8, .01)
    vm = np.arange(-1.2, 1.2, .01)
    lt = np.arange(0, 1.07, .01)
    plt.subplot(2,1,1)
    plt.plot(lm, force_length_muscle(lm), 'r')
    plt.plot(lm, force_length_parallel(lm), 'g')
    plt.plot(lt, force_length_tendon(lt), 'b')
    plt.legend(('CE', 'PE', 'SE'))
    plt.xlabel('Normalized length')
    plt.ylabel('Force scale factor')
    plt.subplot(2, 1, 2)
    plt.plot(vm, force_velocity_muscle(vm), 'k')
    plt.xlabel('Normalized CE velocity')
    plt.ylabel('Force scale factor')
    plt.tight_layout()
    plt.show()


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/2/self.sigma**2)


class Sigmoid:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return expit((x-self.mu) / self.sigma)


class Regression():
    """
    1D regression model with Gaussian basis functions.
    """

    def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
        """
        :param x: samples of an independent variable
        :param t: corresponding samples of a dependent variable
        :param centres: a vector of Gaussian centres (should have similar range of values as x)
        :param width: sigma parameter of Gaussians
        :param regularization_weight: regularization strength parameter
        """
        if sigmoids:
            self.basis_functions = [Sigmoid(centre, width) for centre in centres]
        else:
            self.basis_functions = [Gaussian(centre, width) for centre in centres]
        self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
        self.ridge.fit(self._get_features(x), t)

    def eval(self, x):
        """
        :param x: a new (or multiple samples) of the independent variable
        :return: the value of the curve at x
        """
        return self.ridge.predict(self._get_features(x))

    def _get_features(self, x):
        if not isinstance(x, collections.Sized):
            x = [x]

        phi = np.zeros((len(x), len(self.basis_functions)))
        for i, basis_function in enumerate(self.basis_functions):
            phi[:,i] = basis_function(x)
        return phi


def get_shank_angle_foot_drop():
    data = np.genfromtxt('foot_drop_angular_shank.csv', delimiter=',')  # Data from WebPlotDigitizer extracted into data.csv
    time = data[:, 0]
    angle = data[:, 1]

    # plt.figure()
    # plt.title('Drop Foot Angular Shank')
    # plt.plot(time, angle, 'g')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ankle Angle (deg)")
    # plt.show()

    centres = np.arange(min(time) + 0.305, max(time), .1)
    width = .15
    result = Regression(time, angle, centres, width, .1, sigmoids=False)
    return result, time, angle


def get_shank_angle_normal_foot():
    data = np.genfromtxt('normal_foot_angular_shank.csv', delimiter=',')  # Data from WebPlotDigitizer extracted into data.csv
    angle = data[:, 1]
    time = data[:, 0]

    # plt.figure()
    # plt.title('Normal Foot Angular Shank')
    # plt.plot(time, angle, 'g')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ankle Angle (deg)")
    # plt.show()

    centres = np.arange(min(time) + 0.305, max(time), .1)
    width = .15
    result = Regression(time, angle, centres, width, .1, sigmoids=False)

    return result, time, angle


def get_shank_angle():

    data = np.genfromtxt('Shank.csv', delimiter=',')  # Data from WebPlotDigitizer extracted into data.csv
    angle = data[:, 1]
    time = data[:, 0]

    # plt.figure()
    # plt.title('Normal Foot Angular Shank')
    # plt.plot(time, angle, 'g')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ankle Angle (deg)")
    # plt.show()

    centres = np.arange(min(time), max(time), .01)
    width = .15
    result = Regression(time, angle, centres, width, .1, sigmoids=False)

    return result, time, angle


def get_shank_angle_comp():

    data = np.genfromtxt('Foot_Angle_Comp.csv', delimiter=',')  # Data from WebPlotDigitizer extracted into data.csv
    time = data[:, 0]
    angle = data[:, 1]

    # plt.figure()
    # plt.title('Drop Foot Angular Shank')
    # plt.plot(time, angle, 'g')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Ankle Angle (deg)")
    # plt.show()

    centres = np.arange(min(time), max(time), .01)
    width = .15
    result = Regression(time, angle, centres, width, .1, sigmoids=False)
    return result, time, angle


def regression_error(predicted, label):
    return np.sqrt(np.mean((predicted-label)**2))


def get_muscle_force_velocity_regression():
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:,0]
    force = data[:,1]

    centres = np.arange(-1, 0, .2)
    width = .15
    result = Regression(velocity, force, centres, width, .1, sigmoids=True)

    return result


def get_muscle_force_length_regression():
    """
    CE force-length data samples from Winters et al. (2011) Figure 3C,
    normalized so that max force is ~1 and length at max force is ~1.
    The sampples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.

    WRITE CODE HERE 1) Use WebPlotDigitizer to extract force-length points
    from Winters et al. (2011) Figure 3C, which is on Learn. Click
    "View Data", select all, cut, and paste below. 2) Normalize the data
    so optimal length = 1 and peak = 1. 3) Return a Regression object that
    uses Gaussian basis functions. 
    """
    data = np.genfromtxt('data.csv', delimiter=',')  # Data from WebPlotDigitizer extracted into data.csv
    i_max = data[:, 1].argmax()
    length = data[:, 0] / data[i_max, 0]
    force = data[:, 1] / data[i_max, 1]

    centres = np.arange(min(length) + 0.305, max(length), .1)
    width = .15
    result = Regression(length, force, centres, width, .1, sigmoids=False)
    # plt.scatter(length, force)
    # lm = np.arange(0, 1.8, .01)
    # plt.plot(lm, result.eval(lm))
    # plt.show()

    return result



force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()


def force_length_muscle(lm):
    """
    :param lm: muscle (contracile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)


def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))


def isometric_contraction(f0_max, resting_length_m, resting_length_t, dt=0.01):
    hill_type_muscle = HillTypeMuscle(f0_max, resting_length_m, resting_length_t)
    y0 = [1] #initial state of lm
    t_span = (0, 2)
    L = resting_length_t + resting_length_m

    # simulate lm
    def get_velocity_iso(t, x):
        if t < 0.5:
            a = 0
        else:
            a = 1
        return get_velocity(a, x, hill_type_muscle.norm_tendon_length(L, x))

    sol = solve_ivp(get_velocity_iso, t_span, y0, method="RK45", max_step=dt)
    lm = sol.y.T
    time = sol.t
    force = hill_type_muscle.get_force(L, lm)
    plt.subplot(2, 1, 1)
    plt.title('Isometric Contraction (time step {}s)'.format(dt))
    plt.plot(time, lm, 'b')
    plt.ylabel('Normalized CE length')
    plt.xlabel('Time (s)')
    plt.subplot(2, 1, 2)
    plt.plot(time, force, 'b')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model_drop_foot, time_drop_foot, angle_drop_foot = get_shank_angle_foot_drop()
    model_norm_foot, time_norm_foot, angle_norm_foot = get_shank_angle_normal_foot()
    time2 = np.arange(0, 10, 0.001)
    time = np.arange(0, 6, 0.01)

    plt.figure()
    plt.title('Foot Drop Angular Shank')
    plt.plot(time, model_drop_foot.eval(time), 'b')
    plt.scatter(time, scipy.misc.derivative(model_drop_foot.eval, time, 0.01))
    plt.xlabel("Time (s)")
    plt.ylabel("Ankle Angle (deg)")
    plt.show()

    print('Drop Foot: ', regression_error(model_drop_foot.eval(time_drop_foot), angle_drop_foot))
    _, _, r_value, p_value, _ = scipy.stats.linregress(model_drop_foot.eval(time_drop_foot), angle_drop_foot)
    print('Drop Foot R squared Value: ', r_value ** 2)

    plt.figure()
    plt.title('Normal Foot Angular Shank')
    plt.plot(time2, model_norm_foot.eval(time2), 'g')
    plt.plot(time2, scipy.misc.derivative(model_norm_foot.eval, time2, 0.001))
    plt.xlabel("Time (s)")
    plt.ylabel("Ankle Angle (deg)")
    plt.show()

    print('Normal Foot: ', regression_error(model_norm_foot.eval(time_norm_foot), angle_norm_foot))

    _, _, r_value, p_value, _ = scipy.stats.linregress(model_norm_foot.eval(time_norm_foot), angle_norm_foot)
    print('Normal Foot R squared Value: ', r_value ** 2)




