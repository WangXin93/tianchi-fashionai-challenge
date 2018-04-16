import time
import copy
import torch
from torch.autograd import Variable
from .metric import calculate_ap
import ipdb


def train_model(model,
                criterion,
                optimizer,
                scheduler,
                dataloaders,
                dataset_sizes,
                use_gpu,
                save_file,
                num_epochs=2,
                verbose=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_ap = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            AP = 0.0
            AP_cnt = 0

            batch_num = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs = data['image']
                labels = data['category']

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # inception_v3
#                if phase == 'train':
#                    outputs, aux = model(inputs)
#                else:
#                    outputs = model(inputs)
                outputs = model(inputs)

                # Get softmax scores
                probs = torch.nn.functional.softmax(outputs)
                # Convert to cpu
                probs = probs.data.cpu().numpy()
                ap, cnt = calculate_ap(labels, probs)
                AP += ap
                AP_cnt += cnt

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    if batch_num % 20 == 0 and verbose:
                        print('batch: #{}, loss = {}'.format(batch_num, loss.data[0]))
                    batch_num += 1

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_ap = AP / AP_cnt

            print('{} Loss: {:.4f} Acc: {:.4f} AP: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_ap))

            # deep copy the model
            if phase == 'test' and epoch_ap > best_ap:
                best_ap = epoch_ap
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save model
                torch.save(model.state_dict(), save_file)
                print('Saved to {}'.format(save_file))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def train_model_noval(model,
                criterion,
                optimizer,
                scheduler,
                dataloaders,
                dataset_sizes,
                use_gpu,
                num_epochs=2):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs = data['image']
                labels = data['category']

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                # Inception_v3
                if phase == 'train':
                    outputs, aux = model(inputs)
                else: 
                    outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
#            if phase == 'test' and epoch_acc > best_acc:
#                best_acc = epoch_acc
#                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#    print('Best test Acc: {:4f}'.format(best_acc))
#
#    # load best model weights
#    model.load_state_dict(best_model_wts)
    return model
