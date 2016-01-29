import numpy

def center_frob(x):
    x_mean = x.mean(axis=0)
    x_frob = numpy.diag(1./numpy.sqrt(numpy.sum(x**2, axis=0)))
    x = x.dot(x_frob)
    return x, x_frob

def center_log(x, axis=0):
    x_mean = x.mean(axis=axis)
    x_std = x.std(axis=axis)
    x = (numpy.log(x + 1e-10) - x_mean) / x_std
    return x, x_mean, x_std

def center_boxcox(x):
    from scipy.stats import boxcox
    print "Number of elements less than O: %i" % ((x+1e-10) <= 0).sum()
    if len(x.shape) > 1:
        lmbda = numpy.zeros(x.shape[1])
        for j in range(x.shape[1]):
            x[:,j], lmbda[j] = boxcox(x[:,j]+1e-10)
    elif len(x.shape) == 1:
        x, lmbda = boxcox(x + 1e-10)
    return x, lmbda

if __name__ == "__main__":
    import numpy
    x = numpy.random.uniform(size = (10,100))
    y, l = center_boxcox(x)
 
