import numpy as np
import torch
import tqdm
torch.set_printoptions(linewidth=120)
import time

# 学习率为0.001是会梯度爆炸，1e-4可以跑
def train_step(model, optimizer, criterion, inputs, targets, single_input,
               mixed_precision_enabled, scaler=None, batch_idx=None, gradient_acc_steps=1, metrics=None):
    # Calculate predictions
    with torch.cuda.amp.autocast(enabled=mixed_precision_enabled):
        # outputs, aux_outputs = model(inputs.float())
        outputs = model(inputs.float())
        # print(outputs)
        if single_input:
            outputs = [outputs[0][:1], outputs[1][:1]]
        outputs = torch.cat(outputs, axis=1)
        # if single_input:
        #     aux_outputs = [aux_outputs[0][:1], aux_outputs[1][:1]]
        # aux_outputs = torch.cat(aux_outputs, axis=1)
        # lr = optimizer.state_dict()['param_groups'][0]['lr']
        # Calculate losses
        loss_main = criterion(outputs, targets)
        #print(targets)
        # loss_aux = criterion(aux_outputs, targets)--
        # loss = loss_main + 0.4 * loss_aux
        # a = 0.001
        # loss = loss_main.add(a)
        loss = loss_main
        #acc = (torch.abs(targets - outputs)) / targets
        #print(acc)
        # print(acc)
    # Backpropagate losses
    # print(loss)
    # loss.require_grad(True)
    if mixed_precision_enabled:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=111111111111111)

    # Apply updates
    if (batch_idx + 1) % gradient_acc_steps == 0:
        if mixed_precision_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    if metrics is not None:
        metrics.update(outputs, targets)
    return loss

# @torch.no_grad()
def eval_step(model, criterion, inputs, targets, single_input, metrics=None):
    model.eval()
    # Calculate predictions
    with torch.no_grad():
        outputs = model(inputs.float())
    if single_input:
        outputs = [outputs[0][:1], outputs[1][:1]]
    outputs = torch.cat(outputs, axis=1)
    # Calculate loss
    loss = criterion(outputs, targets)
    if metrics is not None:
        metrics.update(outputs, targets)
    print(outputs)
    print(targets)
    print("============================")
    return loss


def run_epoch(model, criterion, dataloader, device, phase, mixed_precision_enabled,
              optimizer=None, scaler=None, lr_scheduler=None, gradient_acc_steps=None, lr_scheduler_metric='val_loss',
              task_list=('calorie', 'mass', 'fat', 'carb', 'protein'), metrics=None):
    running_loss = 0.0
    # batch_idx = 0
    # print(type(dataloader.dataset))
    # print(dataloader.__len__())
    for batch_idx, batch in enumerate(dataloader):
        # start_time = time.time()
        inputs = batch['image']
        # print("`````````````````````````````````````````````````````")
        # print(inputs)
        # print("`````````````````````````````````````````````````````")
        # print(batch['depth_image'])
        # print(".....................................depth_image")
        target_list = []
        for task in task_list:
            target_list.append(batch[task])

        single_input = inputs.shape[0] == 1
        # Training will not work with bs == 1, so we do a 'hack'
        if single_input:
            print("# Training will not work with bs == 1, so we do a 'hack'")
            dummy_tensor = torch.zeros(batch['image'][:1].shape)
            inputs = torch.cat([batch['image'][:1], dummy_tensor], axis=0)

        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Calculate actual targets
            targets = torch.squeeze(torch.cat(target_list, axis=1))
            if len(targets.shape) == 1:
                targets = torch.unsqueeze(targets, 0)

            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            if phase == 'train':
                loss = train_step(model, optimizer, criterion, inputs,
                                                            targets, single_input,
                                                            mixed_precision_enabled, scaler=scaler,
                                                            batch_idx=batch_idx,
                                                            gradient_acc_steps=gradient_acc_steps, metrics=metrics)
            else:
                loss = eval_step(model, criterion, inputs, targets,
                                                           single_input, metrics=metrics)

        # statistics
        current_loss = loss.item() * inputs.size(0)
        running_loss += current_loss
        # end_time = time.time()
        # inference_time = end_time - start_time
        # print(inference_time)


    if (lr_scheduler_metric == 'val_loss' and phase == 'val') or (
            lr_scheduler_metric == 'train_loss' and phase == 'train'):
        lr_scheduler.step(running_loss)
    # print("lr = {}".format(lr_scheduler.get_last_lr()))
    results = {
        'average loss': running_loss / len(dataloader.dataset)
    }
    # print(results)

    return results

def train_step_ensemble(model, model2, optimizer, optimizer2, criterion, inputs, targets, single_input,
               mixed_precision_enabled, batch_idx=None, gradient_acc_steps=1, metrics=None, devices=None):
    # Calculate predictions
    with torch.cuda.amp.autocast(enabled=mixed_precision_enabled):
        outputs1 = model(inputs.float())
        if single_input:
            outputs1 = [outputs1[0][:1], outputs1[1][:1]]
        outputs1 = torch.cat(outputs1, axis=1)
        loss = criterion(outputs1, targets)
        # loss = loss_main

        outputs2 = model2(inputs.float().to(devices[1]))
        # print(type(outputs2))
        outputs2 = torch.cat(outputs2, axis=1)
        loss2 = criterion(outputs2, targets.to(devices[1]))

        avg_loss = (loss + loss2.to(devices[0])) /2
        outputs = (outputs1 + outputs2.to(devices[0])) / 2
    loss.backward()
    loss2.backward()
    # print(outputs)
    optimizer.step()
    optimizer.zero_grad()

    optimizer2.step()
    optimizer2.zero_grad()
    if metrics is not None:
        metrics.update(outputs, targets)

    return loss, loss2, avg_loss

def eval_step_ensemble(model, model2, criterion, inputs, targets, single_input, metrics=None, devices=None):
    model.eval()
    model2.eval()
    # Calculate predictions
    with torch.no_grad():
        outputs1 = model(inputs.to(devices[0]))
        outputs2 = model2(inputs.to(devices[1]))

    outputs1 = torch.cat(outputs1, axis=1)
    outputs2 = torch.cat(outputs2, axis=1)

    outputs = (outputs1 + outputs2.to(devices[0])) / 2

    if single_input:
        outputs = [outputs[0][:1], outputs[1][:1]]

    # outputs = torch.cat(outputs, axis=1)
    # Calculate loss
    loss = criterion(outputs, targets)
    if metrics is not None:
        metrics.update(outputs, targets)
    return loss



def run_epoch_ensemble(model, model2, criterion, dataloader, devices, phase, mixed_precision_enabled,
              optimizer=None, optimizer2=None, lr_scheduler=None, gradient_acc_steps=None, lr_scheduler_metric='val_loss',
              task_list=('calorie', 'mass', 'fat', 'carb', 'protein'), metrics=None):
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['image']
        target_list = []
        for task in task_list:
            target_list.append(batch[task])

        single_input = inputs.shape[0] == 1
        if single_input:
            print("# Training will not work with bs == 1, so we do a 'hack'")
            dummy_tensor = torch.zeros(batch['image'][:1].shape)
            inputs = torch.cat([batch['image'][:1], dummy_tensor], axis=0)

        # track history if only in train
        with torch.set_grad_enabled(phase == 'train'):
            # Calculate actual targets
            targets = torch.squeeze(torch.cat(target_list, axis=1))
            if len(targets.shape) == 1:
                targets = torch.unsqueeze(targets, 0)

            inputs = inputs.float().to(devices[0])
            targets = targets.float().to(devices[0])
            if phase == 'train':
                loss, loss2, avg_loss = train_step_ensemble(model, model2, optimizer, optimizer2, criterion, inputs,
                                                            targets, single_input,
                                                            mixed_precision_enabled,
                                                            batch_idx=batch_idx,
                                                            gradient_acc_steps=gradient_acc_steps, metrics=metrics, devices=devices)
            else:
                avg_loss = eval_step_ensemble(model, model2, criterion, inputs, targets,
                                                           single_input, metrics=metrics, devices=devices)

        # statistics
        current_loss = avg_loss.item() * inputs.size(0)
        running_loss += current_loss
        # end_time = time.time()
        # inference_time = end_time - start_time
        # print(inference_time)


    if (lr_scheduler_metric == 'val_loss' and phase == 'val') or (
            lr_scheduler_metric == 'train_loss' and phase == 'train'):
        lr_scheduler.step(running_loss)
    # print("lr = {}".format(lr_scheduler.get_last_lr()))
    results = {
        'average loss': running_loss / len(dataloader.dataset)
    }
    # print(results)

    return results