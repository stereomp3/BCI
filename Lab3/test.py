import numpy as np

# sample = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# print(sample[0:2]+sample[3:5])
valid_num = 7
sample = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
test = np.empty((37, 1, 1))

# print(test)
a = test[37 - valid_num::]
b = test[valid_num::]
c = test[valid_num:]
print(c.shape)
print(test[0])
print(c[0])

print(a.shape)
print(b.shape)
print(a[6])
print(b[29])
print(a[0])
print(b[23])
# print(sample[:min_len])

valid_num = 3
test2 = np.array([[[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]]])
test3 = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
print(test2.shape)
print(test2[valid_num:])
print(test2[:valid_num:])
print(test2[:valid_num:].shape)
print(test3[:valid_num])
print(test3[valid_num:])
print(test2[7::])


