import numpy as np
import torch

torch.set_printoptions(linewidth=120)
val_MAX = [3943.325195, 7974, 106.343002, 844.568604, 147.491821]
K = 10
def antiNormalize(inputs):
    for i in range(5):
        inputs[:, i]= inputs[:, i] * val_MAX / K
    return inputs


# 学习率为0.001是会梯度爆炸，1e-4可以跑
def train_step(model, optimizer, criterion, inputs, targets, single_input,
               mixed_precision_enabled, scaler=None, batch_idx=None, gradient_acc_steps=1, metrics=None):
    # Calculate predictions
    with torch.cuda.amp.autocast(enabled=mixed_precision_enabled):
        # outputs, aux_outputs = model(inputs.float())
        outputs = model(inputs.float())

        if single_input:
            outputs = [outputs[0][:1], outputs[1][:1]]

        outputs = torch.cat(outputs, axis=1)

        loss_main = criterion(outputs, targets)

        loss = loss_main

    # Backpropagate losses
    if mixed_precision_enabled:
        scaler.scale(loss).backward()
    else:
        loss.backward()

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


def eval_step(model, criterion, inputs, targets, single_input, metrics=None):
    # Calculate predictions
    x1, x2, x3, outputs, _, _, _ = model(inputs.float(), True)
    if single_input:
        outputs = [outputs[0][:1], outputs[1][:1]]
    # outputs = torch.cat(outputs, axis=1)
    # Calculate loss

    loss = criterion(outputs, targets)
    if metrics is not None:

        metrics.update(outputs, targets)
    return loss


def run_epoch(model, criterion, dataloader, device, phase, mixed_precision_enabled,
              optimizer=None, scaler=None, lr_scheduler=None, gradient_acc_steps=None, lr_scheduler_metric='val_loss',
              task_list=('calorie', 'mass', 'fat', 'carb', 'protein'), metrics=None):
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        inputs = batch['image']

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
                # loss = train_step(model, optimizer, criterion, inputs,
                #                                             targets, single_input,
                #                                             mixed_precision_enabled, scaler=scaler,
                #                                             batch_idx=batch_idx,
                #                                             gradient_acc_steps=gradient_acc_steps, metrics=metrics)

                optimizer.zero_grad()
                # inputs1 = jigsaw_generator(inputs, 8)
                _, _, _, _, output_1, _, _ = model(inputs, False)
                # print("===============================99 lines")
                # output_1 = torch.cat(output_1, axis=1)
                loss1 = criterion(output_1, targets) * 1
                loss1.backward()
                # print("===============================102 lines")
                # print(loss1.shape)
                # print(loss1)
                optimizer.step()

                # Step 2
                optimizer.zero_grad()
                # inputs2 = jigsaw_generator(inputs, 4)

                _, _, _, _, _, output_2, _, = model(inputs, False)
                # print(output_2.shape)
                # output_2 = torch.cat(output_2, axis=1)
                loss2 = criterion(output_2, targets) * 1
                loss2.backward()
                optimizer.step()
                # print("===============================113 lines")
                # Step 3
                optimizer.zero_grad()
                # inputs3 = jigsaw_generator(inputs, 2)
                _, _, _, _, _, _, output_3 = model(inputs, False)
                # print(output_3.shape)
                # output_3 = torch.cat(output_3, axis=1)
                loss3 = criterion(output_3, targets) * 1
                loss3.backward()
                optimizer.step()
                # print("===============================123 lines")

                optimizer.zero_grad()
                x1, x2, x3, output_concat, _, _, _ = model(inputs, True)
                # print("--------")
                # print(targets)
                # output_concat = torch.cat(output_concat, axis=1)
                # output_concat = output_1[0]

                concat_loss = criterion(output_concat, targets) * 1
                # print(output_concat)
                loss = concat_loss

                for i in range(5):
                    MAPE = torch.sum(torch.abs(output_concat[:, i] - targets[:, i])) / torch.sum(targets[:, i]) / targets.shape[0]
                    print(output_concat[:, i])
                    print("{} images: Task {}'s MAPE is {}".format(targets.shape[0], i, MAPE))

                if metrics is not None:
                    metrics.update(output_concat, targets)
            else:
                loss = eval_step(model, criterion, inputs, targets,
                                                           single_input, metrics=metrics)

        # statistics
        current_loss = loss.item() * inputs.size(0)
        running_loss += current_loss
    if (lr_scheduler_metric == 'val_loss' and phase == 'val') or (
            lr_scheduler_metric == 'train_loss' and phase == 'train'):
        lr_scheduler.step(running_loss)

    results = {
        'average loss': running_loss / len(dataloader.dataset)
    }
    # print(results)

    return results
