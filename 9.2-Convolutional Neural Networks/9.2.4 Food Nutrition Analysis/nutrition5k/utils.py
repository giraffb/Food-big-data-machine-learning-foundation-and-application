import argparse
#
import torch
# from torchmetrics import Metric
#
# val_MAX = [485.67688, 700, 31.68, 71.27211, 37.214157]
def parse_args():
    """ Parse the arguments."""
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', default="/home/stu_6/myproject/nutrition-rgb/configs/config_template.yml", help='Name of the base config file without extension.')
    return parser.parse_args()
#
# val_MAX = [3943.325195, 7974, 106.343002, 844.568604, 147.491821]
# K = 10
# def antiNormalize(inputs):
#     for i in range(5):
#         inputs[:, i]= inputs[:,i] * val_MAX / K
#     return inputs
class Metrics:
    def __init__(self, task_list, device, prediction_threshold):
        self.task_list = task_list
        self.mean_absolute_errors = {}
        self.n5k_relative_mae = {}
        self.my_relative_mae = {}
        self.thresholded_accuracy = {}
        self.PMAE = {}
        for idx, task in enumerate(self.task_list):
            self.mean_absolute_errors[task] = MeanAbsoluteError(idx, device)
            self.n5k_relative_mae[task] = N5kRelativeMAE(idx, device)
            self.my_relative_mae[task] = MeanRelativeError(idx, device)
            self.thresholded_accuracy[task] = ThresholdedAccuracy(idx, device, prediction_threshold)
            self.PMAE[task] = PMAE(idx, device)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for idx, task in enumerate(self.task_list):
            # preds = antiNormalize(preds)
            # target = antiNormalize(target)

            self.mean_absolute_errors[task].update(preds, target)
            self.n5k_relative_mae[task].update(preds, target)
            self.my_relative_mae[task].update(preds, target)
            self.thresholded_accuracy[task].update(preds, target)
            self.PMAE[task].update(preds, target)

    def compute(self):
        metrics = {}
        for idx, task in enumerate(self.task_list):
            metrics['{} mean average error'.format(task)] = self.mean_absolute_errors[task].compute()
            metrics['{} n5k relative mean average error'.format(task)] = self.n5k_relative_mae[task].compute()
            metrics['{} my relative mean average error'.format(task)] = self.my_relative_mae[task].compute()
            metrics['{} thresholded accuracy'.format(task)] = self.thresholded_accuracy[task].compute()
            metrics['{} PMAE'.format(task)] = self.PMAE[task].compute()
        return metrics

    # def reset(self):
    #     i = 0
        # for idx, task in enumerate(self.task_list):
        #     self.mean_absolute_errors[task].reset()
        #     self.n5k_relative_mae[task].reset()
        #     self.PMAE[task].reset()

class MeanAbsoluteError(Metrics):
    def __init__(self, task_idx, device, dist_sync_on_step=False):
        # super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task_idx = task_idx
        self.abs_error = torch.tensor(0).float().to(device)
        self.total = torch.tensor(0).float().to(device)
        # self.add_state("abs_error", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        task_preds = preds[:, self.task_idx]
        task_target = target[:, self.task_idx]

        n_samples = target.numel()
        dividends = torch.abs(task_target)
        dividends[dividends == 0] = 1
        abs_difference = torch.abs(task_preds - task_target)
        self.abs_error += torch.sum(abs_difference)
        self.total += n_samples

    def compute(self):

        return self.abs_error / self.total
               # * val_MAX[self.task_idx]


class MeanRelativeError(Metrics):
    def __init__(self, task_idx, device, dist_sync_on_step=False):
        # super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task_idx = task_idx
        self.relative_error = torch.tensor(0).float().to(device)
        self.total = torch.tensor(0).float().to(device)
        # print(self.relative_error)
        # print(self.total)
        # self.add_state("relative_error", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        task_preds = preds[:, self.task_idx]
        task_target = target[:, self.task_idx]

        n_samples = target.numel()
        dividends = torch.abs(task_target)
        dividends[dividends == 0] = 1
        abs_difference = torch.abs(task_preds - task_target)
        self.relative_error += torch.sum(abs_difference / dividends)
        self.total += n_samples

    def compute(self):
        return self.relative_error / self.total

class N5kRelativeMAE(Metrics):
    def __init__(self, task_idx, device, dist_sync_on_step=False):
        # super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task_idx = task_idx
        self.task_mean = (255.0, 215.0, 12.7, 19.4, 18.0)[task_idx]
        self.error = torch.tensor(0).float().to(device)
        self.total = torch.tensor(0).float().to(device)
        # self.add_state("error", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        n_samples = target.numel()
        self.error += torch.sum(torch.tensor(torch.abs(
            preds[:, self.task_idx] - target[:, self.task_idx]) / self.task_mean))
        self.total += n_samples

    def compute(self):
        return self.error / self.total


class ThresholdedAccuracy(Metrics):
    def __init__(self, task_idx, device, prediction_threshold, dist_sync_on_step=False):
        # super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.task_idx = task_idx
        self.prediction_threshold = prediction_threshold
        self.correct = torch.tensor(0).float().to(device)
        self.total = torch.tensor(0).float().to(device)
        # self.add_state("correct", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        n_samples = target.numel()
        dividends = torch.abs(target[:, self.task_idx])
        dividends[dividends == 0] = 1
        self.correct += torch.sum(torch.tensor((torch.abs(
            preds[:, self.task_idx] - target[:, self.task_idx]) / dividends) < self.prediction_threshold))
        self.total += n_samples

    def compute(self):
        return self.correct / self.total

#  self.add_state 的变量可以指定sum计算方式，在每批样本推理后进行累加
#  update时是单批样本的更新，compute是所有测试样本结束后计算
class PMAE(Metrics):
    def __init__(self, task_idx, device, dist_sync_on_step=False):
        # super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.task_list=('calorie', 'mass', 'fat', 'carb', 'protein')
        self.task_idx = task_idx
        self.MAE = torch.tensor(0).float().to(device)
        self.total = torch.tensor(0).float().to(device)
        # self.add_state("MAE", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0).float().to(device), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        n_samples = target.numel()


        self.MAE += 1.0 / n_samples * torch.sum(torch.abs(
                    preds[:, self.task_idx] - target[:, self.task_idx]))

        self.total += 1.0 / n_samples * torch.sum(target[:, self.task_idx])
        # print(self.MAE / self.total)
        # self.PMAE += (MAE / total)

        # ----------2023.4.6-------------

        # self.MAE += torch.sum(torch.abs(
        #     preds[:, self.task_idx] - target[:, self.task_idx]))
        #
        # self.total = torch.sum(target[:, self.task_idx])
        # # self.PMAE = (MAE / total)
        # print(self.MAE / self.total)

        # print(torch.abs(preds[:, self.task_idx] - target[:, self.task_idx]))
        # print("-------------------------")
        # print(torch.abs(preds[:, self.task_idx] - target[:, self.task_idx])
        #                                          / target[:, self.task_idx])
        # print("========================")
        # self.sumPMAE = torch.sum( torch.abs(preds[:, self.task_idx] - target[:, self.task_idx])
        #                                          / target[:, self.task_idx] ) / n_samples
        # self.total += n_samples

        # print(self.sumPMAE)


        # print("-----------------------task{}'s PMAE is {}---------------------".format(self.task_list[self.task_idx], MAE.cpu()/total.cpu()))
        # print(MAE.)
        # print("------------------------------")
        # print(total)
        # print("+++++++++++++++++++++++++++++++++")
        # self.total += n_samples

        # self.mae += MAE
        # self.total += total
        # dividends = torch.abs(target[:, self.task_idx])
        # dividends[dividends == 0] = 1

    def compute(self):
        # return self.mae / self.total * 100
        # current step PMAE, not global PMAE
        print("\n-----------------------task {} PMAE is {}---------------------".format(self.task_list[self.task_idx], self.MAE / self.total))
        return self.MAE / self.total
#
#
# if __name__ == '__main__':
#     s = 1000
#     r = 0.1
#     cnt = 0
#     while s >= 100:
#         # print(s)
#         s = s - s * r - 0.2
#         cnt = cnt + 1
#         print("s:{}  cnt:{}".format(s,cnt))
