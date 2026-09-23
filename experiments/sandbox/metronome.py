import numpy as np
import matplotlib.pyplot as plt


plt.ion()
K = 3.5  # Spring constant
N = 3
omega = np.random.uniform(1.5, 2.3, size=(1, N))  # Random natural frequency
theta = np.random.uniform(0, 2 * np.pi, size=(1, N))  # Random phase
T = 20

def dtheta_dt(theta, omega):
    return omega + K / N * np.sin(theta - theta.T).sum(axis=1)


if __name__ == "__main__":
    dt = 0.01
    t = np.arange(0, T, dt)

    theta_t = np.zeros((len(t), N))
    for i in range(len(t)):
        theta_t[i] = theta.copy()
        theta += dtheta_dt(theta, omega) * dt


    plt.figure(figsize=(10, 6))
    for i in range(N):
        plt.plot(t, theta_t[:, i], label=f'Metronome {i + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Phase (rad)')
    plt.title('Coupled Metronomes')
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(N):
        plt.plot(t, np.sin(theta_t[:, i]), label=f'Metronome {i + 1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (sin(theta))')
    plt.title('Displacement of Coupled Metronomes')
    plt.grid()

    plt.show()
