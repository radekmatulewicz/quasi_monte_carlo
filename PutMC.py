import numpy as np


def Box_Muller(u1, u2):
    theta = 2 * np.pi * u2
    rho = np.sqrt(-2 * np.log(u1))
    z1 = rho * np.cos(theta)
    z2 = rho * np.sin(theta)
    return z1, z2


def Halton(N, b1, b2):
    w1 = np.array([])
    w2 = np.array([])

    for i in range(1, N + 1):
        n0 = i
        m0 = i
        hn = 0
        hn2 = 0
        f = 1 / b1
        g = 1 / b2

        while n0 > 0:
            n1 = int(n0 / b1)
            r = n0 - n1 * b1
            hn = hn + f * r
            f = f / b1
            n0 = n1

        while m0 > 0:
            n2 = int(m0 / b2)
            q = m0 - n2 * b2
            hn2 = hn2 + g * q
            g = g / b2
            m0 = n2

        w1 = np.append(w1, hn)
        w2 = np.append(w2, hn2)

    return w1, w2


def PutMC(S0, r, sigma, T, K, M):
    price1 = np.array([])  # Wygenerowane z rozkladu jednostajnego
    price2 = np.array([])  # Wygenerowane z ciagu Haltona

    # Generacja liczb pseudolosowych za pomocą funkcji random.uniform:
    for i in range(int(M / 2)):
        u1 = np.random.uniform(0, 1, 1)
        u2 = np.random.uniform(0, 1, 1)
        [W1, W2] = Box_Muller(u1, u2)
        stock_price1 = S0 * np.exp((r - 0.5 * (sigma ** 2)) * T + sigma * np.sqrt(T) * W1)
        stock_price2 = S0 * np.exp((r - 0.5 * (sigma ** 2)) * T + sigma * np.sqrt(T) * W2)
        V1 = np.exp(-r * T) * np.maximum(K - stock_price1, 0)
        V2 = np.exp(-r * T) * np.maximum(K - stock_price2, 0)
        price1 = np.append(price1, [V1, V2])

    # Generacja liczb pseudolosowych za pomocą ciągu Haltona (quasi Monte-Carlo):
    [h1, h2] = Halton(int(M / 2), 2, 3)
    X1 = np.array([])
    X2 = np.array([])
    for i in range(int(M / 2)):
        [x1, x2] = Box_Muller(h1[i], h2[i])
        X1 = np.append(X1, x1)
        X2 = np.append(X2, x2)

    qStock_price1 = S0 * np.exp((r - 0.5 * (sigma ** 2)) * T + sigma * np.sqrt(T) * X1)
    qStock_price2 = S0 * np.exp((r - 0.5 * (sigma ** 2)) * T + sigma * np.sqrt(T) * X2)
    qV1 = np.exp(-r * T) * np.maximum(K - qStock_price1, 0)
    qV2 = np.exp(-r * T) * np.maximum(K - qStock_price2, 0)
    price2 = np.append(qV1, qV2)

    # Przedział wiarygodnosci:
    am = np.sum(price1) / M
    bm2 = 1 / (M - 1) * sum((price1 - am) ** 2)
    P = [am - 1.96 * np.sqrt(bm2) / np.sqrt(M), am + 1.96 * np.sqrt(bm2) / np.sqrt(M)]

    MC = np.sum(price1) / M
    qMC = np.sum(price2) / M

    return MC, qMC, P
