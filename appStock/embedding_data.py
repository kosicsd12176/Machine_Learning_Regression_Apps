import numpy

def embed_data(x, steps):
    n = len(x)
    xout = numpy.zeros((n - steps, steps))
    yout = x[steps:]
    for i in numpy.arange(steps, n):
        xout[i - steps] = x[i-steps:i]
    return xout, yout