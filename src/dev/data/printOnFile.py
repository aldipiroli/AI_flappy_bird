import numpy as np

out_file = open("data.txt","w")

for i in xrange(0,10000):
	x1 = np.random.randint(0, 11)
	x2 = np.random.randint(0, 11)
	y = x1 + x2

	#x1 = np.random.random()
	#x2 = np.random.random()

	#y = np.random.random() 
	
	print(x1,x2,y)

	out_file.write("%f," % x1)
	out_file.write("%f," % x2)
	out_file.write("%f\n" % y)

out_file.close()
