import os
import time
import torch
from torch import nn, optim
from models.LSTM_AE import LSTM_AE
from util import get_train_data
if __name__ == '__main__':
    start = time.time()
    model = LSTM_AE(10,5,2)
    model.to("cuda:0")
    min_loss = 9999999
    inputs = get_train_data()
    inputs = torch.Tensor(inputs).to("cuda:0")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    base_path = "experiment/LSTM_AE/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for epoch in range(1, 1000 + 1):
        model.train()  # 开始训练
        optimizer.zero_grad()
        recon= model(inputs)
        loss = criterion(recon, inputs)
        loss.backward()
        optimizer.step()  # 下一步 运行
        print('epoch %d | loss=%.3f | Learning Rate:%.3f' % (epoch, loss.item(),optimizer.param_groups[0]['lr']))
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model, os.path.join(base_path, "best_net.pkl"))
    run_time = time.time() - start
    run_time = time.gmtime(run_time)
    days = run_time.tm_mday - 1
    hours = run_time.tm_hour
    minutes = run_time.tm_min
    seconds = run_time.tm_sec
    print("time span : {}d{}h{}m{}s".format(days, hours, minutes, seconds))
