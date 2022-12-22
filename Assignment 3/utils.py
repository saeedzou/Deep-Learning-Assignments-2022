'''This file contains utility functions for the assignment.'''
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_metrics(model, test_loader, device='cpu', verbose=True, classes=None):
    """
    Calculate accuracy, error rate, precision, recall and confusion matrix
    for each class.

    Args:
        model: model to evaluate
        test_loader: test data loader
        device: device to use
        verbose: print metrics or not
        classes: list of classes

    Returns:
        accuracy, error_rate, precision, recall, confusion_matrix

    """
    model.eval()
    with torch.no_grad():
        confusion_matrix = torch.zeros(len(classes), len(classes))
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
        if verbose:
            # print(f'Accuracy: {accuracy:.3f}, '
            # f'Error rate: {error_rate:.3f}, '
            # f'Precision: {precision}, '
            # f'Recall: {recall}')
            print("------------------Classification Report------------------")
            print("\t\t precision \t recall \t f1-score \t support")
            for i in range(3):
                print(f'{classes[i]} \t\t {precision[i]:.3f} \t\t {recall[i]:.3f} \t\t '
                f'{2 * precision[i] * recall[i] / (precision[i] + recall[i]):.3f} \t\t '
                f'{confusion_matrix[i].sum().int()}')
            print("--------------------------------------------------------")
            print(f'Accuracy: \t\t\t\t\t {accuracy:.3f} \t\t {len(test_loader.dataset)}')
            print(f'Error rate: \t\t\t\t\t {error_rate:.3f} \t\t {len(test_loader.dataset)}')
            print(f'Macro avg: \t {precision.mean():.3f} \t\t {recall.mean():.3f} \t\t '
            f'{2 * precision.mean() * recall.mean() / (precision.mean() + recall.mean()):.3f} '
            f'\t\t {len(test_loader.dataset)}')
            f1_weighted = 2 * precision.dot(confusion_matrix.sum(1)) / confusion_matrix.sum() * \
                recall.dot(confusion_matrix.sum(0)) / confusion_matrix.sum() / \
                    (precision.dot(confusion_matrix.sum(1)) / confusion_matrix.sum()\
                         + recall.dot(confusion_matrix.sum(0)) / confusion_matrix.sum())
            print('Weighted avg: \t '
            f'{precision.dot(confusion_matrix.sum(1)) / confusion_matrix.sum():.3f} \t\t '
            f'{recall.dot(confusion_matrix.sum(0)) / confusion_matrix.sum():.3f} \t\t '
            f'{f1_weighted:.3f} \t\t {len(test_loader.dataset)}')
            print("--------------------------------------------------------")
            print("\n\n")
            print('------------------Confusion Matrix------------------')
            sns.heatmap(confusion_matrix.int(), annot=True,
            xticklabels=classes, yticklabels=classes, fmt='d', cmap='Blues')
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
