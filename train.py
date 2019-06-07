import caffe
import numpy as np
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.AdamSolver('solver.prototxt')

# solver.solve()
epochs = 100
train_iter_per_epoch = 2307

g_avg_loss = []
g_avg_acc = []
best_acc = 0
for i in range(epochs):
    avg_loss = np.zeros(train_iter_per_epoch)
    avg_acc = np.zeros(train_iter_per_epoch)
    for j in range(train_iter_per_epoch):
        solver.step(1)
        avg_loss[j] = solver.net.blobs['loss'].data
        avg_acc[j] = solver.net.blobs['accuracy'].data
        if j % 50 == 0:
            mean_acc = avg_acc.sum()/(j+1)
            mean_loss = avg_loss.sum()/(j+1)
            g_avg_loss.append(mean_loss)
            g_avg_acc.append(mean_acc)
            print('epoch: %d, iters: %d, loss: %.4f, acc: %.4f, finished: %.2f'%(i+1, i*train_iter_per_epoch+j, mean_loss, mean_acc, 100.0*(j+1)/train_iter_per_epoch))
            # print('epoch: %d, iter: %d, loss: %.4f, acc: %.4f'%(i*1250+j, mean_loss, mean_acc))
    if avg_acc.mean() > best_acc:
        best_acc = avg_acc.mean()
        solver.net.save('results/iter_%d_best_acc=%.4f.caffemodel'%(i+1, best_acc))


with open('results/loss.txt', 'w') as f:
	for val in g_avg_loss:
		f.write("%.4f\n"%val)
with open('results/acc.txt', 'w') as f:
	for val in g_avg_acc:
		f.write("%.4f\n"%val)

import matplotlib.pyplot as plt 
plt.imshow(g_avg_loss)
plt.show()
