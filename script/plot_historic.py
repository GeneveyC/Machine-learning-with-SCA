import numpy as np
import matplotlib.pyplot as plt


f = "../models/avr/zaid/multi_channels/1kfold_traces_O0_3_SNR4_1_EM2/hist_1.npy"
history=np.load(f, allow_pickle='TRUE').item()

plt.figure()
plt.plot(history["loss"], label='loss training')
plt.plot(history["val_loss"], label='loss validation')
plt.show()

plt.figure()
plt.plot(history["mean_rank"], label='rank training')
plt.plot(history["val_mean_rank"], label='rank validation')
plt.show()

plt.figure()
plt.plot(history["accuracy"], label='accuracy training')
plt.plot(history["val_accuracy"], label='accuracy validation')
plt.show()