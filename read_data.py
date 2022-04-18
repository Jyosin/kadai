import numpy as np

a = np.loadtxt('/Users/wangruqin/VScode/kadai1/spiral_data/2class.txt')
x_dataset = np.delete(a,2,axis = 1)
y_dataset = np.delete(a,[0,1],axis = 1)


print(x_dataset)
print(y_dataset)

