import matplotlib.pyplot as plt
import numpy as np

# f= open("lstm.txt","r")
# content = f.readlines()
T_options = list(range(5, 36, 2))
all_accuracies = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.140625, 0.1328125, 0.1328125], [0.0859375, 0.0859375, 0.1015625], [0.1015625, 0.09375, 0.0859375], [0.1328125, 0.109375, 0.1171875], [0.09375, 0.09375, 0.078125], [0.15625, 0.1328125, 0.1328125], [0.0625, 0.0625, 0.0625], [0.140625, 0.1328125, 0.1328125], [0.078125, 0.09375, 0.1015625], [0.1171875, 0.109375, 0.1015625]]
print(all_accuracies)
all_losses = [[0.0012054890394210815, 0.0011258125305175781, 0.001289553940296173], [0.0007829740643501282, 0.0007598698139190674, 0.0006293430924415588], [0.0004276782274246216, 0.00046715885400772095, 0.0006274431943893433], [0.00038032978773117065, 0.00046152621507644653, 0.0005482956767082214], [0.0003321915864944458, 0.0004029795527458191, 0.0003104880452156067], [0.00034496188163757324, 0.00037194788455963135, 0.0003576353192329407], [2.2993106842041016, 2.299198627471924, 2.3003385066986084], [2.305449962615967, 2.3043293952941895, 2.3052093982696533], [2.309819221496582, 2.308500289916992, 2.3109588623046875], [2.2996087074279785, 2.2978601455688477, 2.2966055870056152], [2.3083999156951904, 2.3063483238220215, 2.3063509464263916], [2.3035385608673096, 2.301921844482422, 2.303849458694458], [2.314004421234131, 2.3128538131713867, 2.3114728927612305], [2.2942709922790527, 2.296518087387085, 2.293539524078369], [2.3047728538513184, 2.3028359413146973, 2.304363250732422], [2.3055710792541504, 2.3022255897521973, 2.303934335708618]]
# print(all_losses)
all_train_steps = [[910, 880, 930], [960, 1020, 1060], [1100, 1050, 1070], [1550, 1230, 1140], [1480, 1340, 1340], [1660, 2590, 1450], [501, 501, 501], [501, 501, 501], [501, 501, 501], [501, 501, 501], [501, 501, 501], [501, 501, 501], [501, 501, 501], [501, 501, 501], [501, 501, 501], [501, 501, 501]]
# print(all_train_steps)

plt.figure(figsize=(19, 6))

mean_acc = list(np.array(all_accuracies).mean(axis=1))
maxstd_acc = list(np.array(all_accuracies).mean(axis=1) + np.array(all_accuracies).std(axis=1))
minstd_acc = list(np.array(all_accuracies).mean(axis=1) - np.array(all_accuracies).std(axis=1))

mean_loss = list(np.array(all_losses).mean(axis=1))
maxstd_loss = list(np.array(all_losses).mean(axis=1) + np.array(all_losses).std(axis=1))
minstd_loss = list(np.array(all_losses).mean(axis=1) - np.array(all_losses).std(axis=1))

mean_train_steps = list(np.array(all_train_steps).mean(axis=1))

plt.subplot(1, 3, 1)
plt.plot(T_options, all_accuracies, "o", color="orange")
plt.plot(T_options, mean_acc, label="accuracy mean", color="orange")
plt.fill_between(T_options, maxstd_acc, minstd_acc, alpha = 0.2, label="accuracy std", color="orange")
plt.xlabel("input_length")
plt.ylabel("accuracy")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(T_options, all_losses, "o", color="orange")
plt.plot(T_options, mean_loss, label="loss mean", color="orange")
plt.fill_between(T_options, maxstd_loss, minstd_loss, alpha = 0.2, label="loss std", color="orange")
plt.xlabel("input_length")
plt.ylabel("loss")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(T_options, mean_train_steps, label="average train steps", color="orange")
plt.xlabel("input_length")
plt.ylabel("train steps")
plt.legend()

plt.savefig("ltsm2.png")

plt.show()