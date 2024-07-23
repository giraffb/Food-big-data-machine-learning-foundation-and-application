import torch
from torch import Tensor
import math
import torch.nn.functional as F
import torch.nn as nn

val_MAX = [3943.325195, 7974, 106.343002, 844.568604, 147.491821]
K = 1

def antiNormalize(inputs):
    for i in range(5):
        inputs[:, i] = inputs[:, i] * val_MAX / K
    return inputs

def n5kloss(output: Tensor, target: Tensor, use_macronutrients=True) -> Tensor:
    batch_size = output.shape[0]

    # output = antiNormalize(output)
    # target = antiNormalize(target)
    # inf = 10000
    # for i in range(batch_size):
    #     for j in range(5):
    #         if abs(output[i][j]) > inf:
    #             print(output[i][j])
    #             output[i][j] = 0

    calories_diff = torch.abs(output[:, 0] - target[:, 0])
    total_mass_diff = torch.abs(output[:, 1] - target[:, 1])

    total_loss = calories_diff + total_mass_diff + (torch.abs(output[:, 2] - target[:, 2])\
                    + torch.abs(output[:, 3] - target[:, 3]) + torch.abs(output[:, 4] - target[:, 4]))
#     if use_macronutrients:
# #        print(output)
# #        print(target)
#         macro_diff = torch.abs(output[:, 3:] - target[:, 3:])
#         total_loss += torch.mean(macro_diff, dim=1)
#     total_loss = torch.abs(output[:, :] - target[:, :])
#     total_loss = torch.
#     print(total_loss)
#     print(output[:3,:])
#     print(target[:3,:])
#     print(output)
#     print("===================\n")

    return torch.sum(total_loss) / batch_size


def swinNLoss(output: Tensor, target: Tensor) -> Tensor:
    batch_size = output.shape[0]
    Loss = 0
    # print("+++++++++++++++++++")
    # print(output[:10, 0] * 485.67688)
    # print("___________________")
    # print(output)
    # print(target)
    for i in range(5):
        # get every task's certainty
        sigma2 = torch.sum(torch.abs((output[:, i] - target[:, i]) ** 2)) / batch_size
        # print(sigma2)
        # 计算每个任务的MSE损失
        loss = torch.sum(torch.abs(target[:, i] - output[:, i])) # shape为(num_tasks,)

        Loss += 1.0 / batch_size * loss / (2 * sigma2) + torch.log(sigma2) / 2
    # print(output)
    # print("===================\n\n\n")
    # print(target)
    # print(LT)
    # print(Loss)
    return Loss




#
# def multi_task_loss(output, target, uncertainty):
#     loss = 0
#     for i in range(output.shape[1]):
#         weighted_error = torch.abs(output[:,i] - target[:,i]) / uncertainty[:,i]
#         loss += torch.mean(weighted_error)
#     return loss
#
#
# def evaluate_uncertainty(model, dataloader, num_samples=10):
#     # 评估每个任务的不确定度
#     model.eval()
#     with torch.no_grad():
#         uncertainty = torch.zeros((len(dataloader.dataset), num_tasks))
#         for i, (inputs, targets) in enumerate(dataloader):
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#             for j in range(num_samples):
#                 outputs = model(inputs)  # 在输入数据上进行多次预测
#                 uncertainty[i * num_samples + j] += torch.var(outputs, dim=0)
#         uncertainty /= num_samples
#     return uncertainty


