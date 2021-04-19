acc = []  # 获取验证集准确性数据
# val_acc=list(map(lambda x:x,val_acc))
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Trainning acc')  # 以epochs为横坐标，以训练集准确性为纵坐标
plt.plot(epochs, val_acc, 'b', label='Vaildation acc')  # 以epochs为横坐标，以验证集准确性为纵坐标
plt.legend()  # 绘制图例，即标明图中的线段代表何种含义

plt.figure()  # 创建一个新的图表
plt.plot(epochs, loss, 'bo', label='Trainning loss')
plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
plt.legend()  ##绘制图例，即标明图中的线段代表何种含义

plt.show()  # 显示所有图表