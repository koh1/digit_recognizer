import numpy
import theano
import theano.tensor as T

#W = theano.shared(
#    value = numpy.zeros(
#        (n_in, n_out),
#        dtype=theano.config.floatX
#    ),
#    name = 'W',
#    borrow = True
#)

x = numpy.array([1,1,1])
xs = theano.shared(x)

print(xs)

print(type(xs))

print(xs.get_value())

x[1] = 2
print(x)

print(xs.get_value())

xs = theano.shared(x, borrow=True)

print(xs.get_value())
x[1] = 3
print(x)
print(xs.get_value())

