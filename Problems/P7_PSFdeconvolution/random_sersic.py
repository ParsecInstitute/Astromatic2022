import numpy as np

def draw_random_sersic(shape, n = [0.5, 5], logRe = [0, 2], logIe = [-1, 2], q = [0.1, 0.9], PA = [0, np.pi]):

    center = np.random.uniform(np.zeros(2), shape, size = 2)

    n = np.random.uniform(n[0], n[1])

    Re = 10**np.random.uniform(logRe[0], logRe[1])

    Ie = 10**np.random.uniform(logIe[0], logIe[1])

    q = np.random.uniform(q[0], q[1])

    PA = np.random.uniform(PA[0], PA[1])
    
    XX, YY = np.meshgrid(np.arange(shape[0]) - center[0], np.arange(shape[1]) - center[1])

    XX, YY = XX*np.cos(PA) - YY*np.sin(PA), XX*np.sin(PA) + YY*np.cos(PA)
    
    RR = np.sqrt(XX**2 + (YY / q)**2)

    bn = 2*n - 1/3
    return Ie * np.exp(-bn*((RR/Re)**(1/n) - 1))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.imshow(np.log10(draw_random_sersic((1000,1000))), origin = "lower")
    plt.show()
