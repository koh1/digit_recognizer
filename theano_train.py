
# coding: utf-8

# In[4]:

from theano import function,config,shared,sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f =  function([], T.exp(x))
print f.maker.fgraph.toposort()
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1-t0))
print("Result is %s" % r)

if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print("used the CPU")
else:
    print("used the GPU")


# In[ ]:



