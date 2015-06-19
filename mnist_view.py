import cPickle, gzip, numpy, sys

if len(sys.argv) != 2:
  quit()
data_index=int(sys.argv[1])
f=gzip.open('./data/mnist.pkl.gz','rb')
train_set, valid_set, test_set=cPickle.load(f)
train_set_x, train_set_y=train_set 
for i in range(data_index,data_index+1):
    for y in range(0,28):
        for x in range(0,28):
            if train_set_x[i][y*28+x]<0.5:
                sys.stdout.write(" ")
            elif train_set_x[i][y*28+x]<0.8:
                sys.stdout.write("+")
            else:
                sys.stdout.write("*")
        sys.stdout.write("\n")
print "correct =",train_set_y[i]
print "--------------------------------"

