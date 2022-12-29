'''This file contains utility functions for the assignment.'''
import torch
import seaborn as sns
import torch.nn.functional as F


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer,
                device, writer=None, verbose=True, teacher_model=None,
                alpha=0.5, T=10):
    """
    This function trains the model for the given number of epochs and returns
    loss and accuracy of train and val sets.

    Args:
    model: model to train
    train_loader: train data loader
    val_loader: validation data loader
    epochs: number of epochs to train
    criterion: loss function
    optimizer: optimizer
    device: device to use for training (cpu or gpu)
    tensorboard: if True, add loss and accuracy to tensorboard
    writer: tensorboard writer
    verbose: print loss and accuracy for each epoch or not
    teacher_model: teacher model for distillation
    alpha: weight for distillation loss
    T: temperature for distillation loss

    Returns:
    train_losses: list of train losses for each epoch
    val_losses: list of val losses for each epoch
    train_acc: list of train accuracies for each epoch
    val_acc: list of val accuracies for each epoch

    """
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        train_correct = 0
        val_correct = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(images)
                loss = criterion(output, labels) * alpha + \
                    criterion(F.log_softmax(output / T, dim=1),
                    F.softmax(teacher_output / T, dim=1)) * (1 - alpha) * T ** 2
            else:
                loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(labels.view_as(pred)).sum().item()
        train_losses.append(train_loss / len(train_loader))
        train_acc.append(train_correct / len(train_loader.dataset))
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_output = teacher_model(images)
                    loss = criterion(output, labels) * alpha + \
                        criterion(F.log_softmax(output / T, dim=1),
                        F.softmax(teacher_output / T, dim=1)) * (1 - alpha) * T ** 2
                else:
                    loss = criterion(output, labels)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(labels.view_as(pred)).sum().item()
        val_losses.append(val_loss / len(val_loader))
        val_acc.append(val_correct / len(val_loader.dataset))
        if verbose:
            print("Epoch: {}/{} \t Train loss: {:.3f} \t "
                  "Train accuracy: {:.3f} \t val loss: {:.3f} "
                  "\t val accuracy: {:.3f}"
                  .format(epoch + 1, epochs, train_loss / len(train_loader),
                          train_correct / len(train_loader.dataset),
                          val_loss / len(val_loader),
                          val_correct / len(val_loader.dataset)))
        if writer is not None:
            writer.add_scalar('Loss/train',
                              train_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/val',
                              val_loss / len(val_loader), epoch)
            writer.add_scalar('Accuracy/train',
                              train_correct / len(train_loader.dataset), epoch)
            writer.add_scalar('Accuracy/val',
                              val_correct / len(val_loader.dataset), epoch)
            writer.close()
    return train_losses, val_losses, train_acc, val_acc

def print_class_metrics(metrics):
    """
    This function prints the metrics for each class and the average metrics.

    Args:
    metrics: dictionary of metrics for each class

    """
    print("---------------------------Classification Report---------------------------")
    print("\t\t precision \t recall \t f1-score \t support")
    for i in metrics:
        print(i, "\t", metrics[i]['precision'], "\t", metrics[i]['recall'], "\t", metrics[i]['f1-score'], "\t", int(metrics[i]['support']))
    print("--------------------------------------------------------------------------")
    # calculate and print the average metrics
    avg_precision = sum([metrics[i]['precision'] for i in metrics])/len(metrics)
    avg_recall = sum([metrics[i]['recall'] for i in metrics])/len(metrics)
    avg_f1 = sum([metrics[i]['f1-score'] for i in metrics])/len(metrics)
    total_support = sum([int(metrics[i]['support']) for i in metrics])
    print(f"macro avg \t {avg_precision:8f} \t {avg_recall:8f} \t {avg_f1:8f} \t {int(total_support)}")
    # calculate and print the weighted average metrics
    weighted_precision = sum([metrics[i]['precision']*int(metrics[i]['support']) for i in metrics])/total_support
    weighted_recall = sum([metrics[i]['recall']*int(metrics[i]['support']) for i in metrics])/total_support
    weighted_f1 = sum([metrics[i]['f1-score']*int(metrics[i]['support']) for i in metrics])/total_support
    print(f"weighted avg \t {weighted_precision:8f} \t {weighted_recall:8f} \t {weighted_f1:8f} \t {int(total_support)}")


def calculate_metrics(model, test_loader, classes, device='cpu', verbose=True):
    """
    Calculate accuracy, error rate, precision, recall and confusion matrix
    for each class.

    Args:
        model: model to evaluate
        test_loader: test data loader
        device: device to use
        verbose: print metrics or not
        classes: list of classesss

    Returns:
        accuracy, error_rate, precision, recall, confusion_matrix

    """
    model.eval()
    with torch.no_grad():
        confusion_matrix = torch.zeros(len(classes), len(classes))
        class_metrics = {i: {'precision': [], 'recall': [], 'f1-score': [], 'support': []} for i in classes}
        test_correct = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(labels.view_as(pred)).sum().item()
            for i, j in zip(labels.view(-1), pred.view(-1)):
                confusion_matrix[i.long(), j.long()] += 1
        accuracy = test_correct / len(test_loader.dataset)
        error_rate = 1 - accuracy
        precision = (confusion_matrix.diag() / confusion_matrix.sum(1)).numpy()
        recall = (confusion_matrix.diag() / confusion_matrix.sum(0)).numpy()
        f1_score = 2 * precision * recall / (precision + recall)
        support = confusion_matrix.sum(1).numpy()
        for (i, j) in enumerate(classes):
            class_metrics[j]['precision'] = precision[i]
            class_metrics[j]['recall'] = recall[i]
            class_metrics[j]['f1-score'] = f1_score[i]
            class_metrics[j]['support'] = support[i]
        if verbose:
            print_class_metrics(class_metrics)
            print('------------------Accuracy and Error Rate------------------')
            print(f"Accuracy: {accuracy*100:.2f}%")
            print(f"Error rate: {error_rate*100:.2f}%")
            print('------------------Confusion Matrix------------------')
            sns.heatmap(confusion_matrix.int(), annot=True,
                        xticklabels=classes, yticklabels=classes, fmt='d',
                        cmap='Blues')
        else:
            return accuracy, error_rate, precision, recall, confusion_matrix


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: model to evaluate

    Returns:
        number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
