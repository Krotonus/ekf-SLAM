import numpy as np

def predict(mu, cov, u, Rt):
    n = len(mu)

    [dtrans, drot1, drot2] = u
    motion = np.array([[dtrans * np.cos(mu[2][0] + drot1)],
                       [dtrans * np.sin(mu[2][0] + drot1)],
                       [drot1 + drot2]])
    F = np.append(np.eye(3), np.zeros((3, n-3)), axis = 1)

    mu_bar = mu + (F.T).dot(motion)

    J = np.array([[0, 0, -dtrans * np.sin(mu[2][0] + drot1)],
                  [0, 0, dtrans * np.cos(mu[2][0] + drot1)],
                  [0, 0, 0]])

    G = np.eye(n) + (F.T).dot(J).dot(F)

    cov_bar = G.dot(cov).dot(G.T) + (F.T).dot(Rt).dot(F)

    print("Predicted location \tx: {0:.2f} \t y: {1:.2f} \t theta: {2:.2f}".format(mu_bar[0][0], mu_bar[1][0], mu_bar[2][0]))

    return mu_bar, cov_bar

def update(mu, cov, obs, c_prob, Qt):
    N = len(mu)

    for [r, theta, j] in obs:
        j = int(j)
        if cov[2*j+3][2*j+3] >= 1e6 and cov[2*j+4][2*j+4] >= 1e6:
            mu[2*j+3][0] = mu[0][0] + r*np.cos(theta+mu[2][0])
            mu[2*j+4][0] = mu[1][0] + r*np.sin(theta+mu[2][0])


