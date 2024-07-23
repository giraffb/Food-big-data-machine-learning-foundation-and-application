import time

import torch

torch.set_printoptions(linewidth=120)
import yaml

from nutrition5k.swinFFM import Swin_Nutrition
from nutrition5k.utils import parse_args, Metrics
from PIL import Image
from torchvision.transforms import transforms
from nutrition5k.nutritionModel import create_model
from nutrition5k.train_utils import eval_step
from nutrition5k import n5kloss
# from nutrition5k.dataset import Resize,ToTensor,Normalize
transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224,224)),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    # inputs = Image.open("/home/NLM/myProgram/nutrition5k-main/dataset/side_angles/dish_1562690917/camera_B/14.jpg")
    path = "/home/stu_6/myproject/nutrition-rgb/runs/inception_v4_nutrition/2023-10-17 14:33:43.360851/epoch_97/state_dict97.pth"
    with open(args.config_path, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = torch.load('/home/stu_6/myproject/nutrition-rgb/runs/inception_v4_nutrition/2023-10-02 10:42:04.420860/dataloaders.pt')
    state_dict = torch.load(path, map_location=device)
    model = create_model(backone="inception_v4", tasks=config["task_list"], device=device, fc_d=2048)

    task_list = config['task_list']
    metrics = Metrics(config['task_list'], device, config['prediction_threshold'])
    criterion = n5kloss
    # Load a checkpoint
    model.load_state_dict(state_dict['net'])
    model = model.eval()
    for batch_idx, batch in enumerate(dataloader['val']):
        inputs = batch['image']
        target_list = []
        for task in task_list:
            target_list.append(batch[task])
        targets = torch.squeeze(torch.cat(target_list, axis=1))
        if len(targets.shape) == 1:
            targets = torch.unsqueeze(targets, 0)

        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        loss = eval_step(model, criterion, inputs, targets, 0, metrics=metrics)

    metrics.compute()

    # model.eval()
    # # inputs.to("cuda")
    # output = model(inputs.float().to("cuda"))
    # print(output)
    # since = time.time()
    # model.eval()
    # results = run_epoch(model, criterion, dataloader, device, 'test', False, config['prediction_threshold'])
    # time_elapsed = time.time() - since
    # print('{} loss: {:.4f}'.format('Test', results['average loss']))
    # print('{} mass prediction accuracy: {:.4f}'.format('Test', results['mass prediction accuracy']))
    # print('{} calorie prediction accuracy: {:.4f}'.format('Test', results['calories prediction accuracy']))
    # print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
