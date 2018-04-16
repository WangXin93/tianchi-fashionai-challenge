import numpy as np


def evaluate(preds, labels, threshold=0.0, verbose=True):
    """Evaluate predictions based on original labels

    Args:
        preds (numpy.ndarray): A list of predictions in 2-D shape,
            with first dimension as number of samples, second is 
            categories of classes
        labels: (list) A list of strings like 'nynnnm'

    Return:
        Accuracy(float): pred_correct_count / pred_count
    """
    if verbose:
        print('Start evaluate %s samples...' % len(preds))
    block_count = 0
    pred_count = 0
    pred_correct_count = 0
    for pred, label in zip(preds, labels):
        if max(pred) < threshold:
            # While max score small than threshold, skip this sample
            block_count += 1
            continue
        elif max(pred) >= threshold:
            if label[np.argmax(pred)] == 'y':
                pred_count += 1
                pred_correct_count += 1
            elif label[np.argmax(pred)] == 'm':
                pass
            elif label[np.argmax(pred)] == 'n':
                pred_count += 1
    if verbose:
        print('%s samples are blocked' % block_count)
        print('%s / %s samples are predicted correctly.' %
              (pred_correct_count, pred_count))
    return pred_correct_count / pred_count


def AP(preds, labels):
    """Get average evaluate result for each threshold

    Args:
        preds: 2-D array with element range from 0 to 1.
        labels: list of string like 'nnnnmy'.

    Returns:
        a float value called AP means avarage value of evaluate
        result of threshold from 0 to 1.
    """
    outs = []
    thredholds = np.max(preds, axis=1)
    for threshold in thredholds:
        out = evaluate(preds, labels, threshold, verbose=False)
        outs.append(out)
    return sum(outs) / len(outs)
        

def mAP(preds_list, labels_list):
    """Get mAP for 8 attributes

    Args:
        preds_list: list of 8 attributes predictions.
        labels_list: list of 8 attributes labels.

    Return:
        a float value called mAP means avarage value of 8 AP of 
        different attributes.
    """
    aps = []
    for preds, labels in zip(preds_list, labels_list):
        ap = AP(preds, labels)
        aps.append(ap)
    return sum(aps) / len(aps)


def calculate_ap(labels, outputs):
    """calculate ap of batch samples

    # https://github.com/hetong007/Gluon-FashionAI-Attributes/blob/master/FashionAI-Attributes-Skirt.ipynb

    Args:
        labels: list of ground-truth labels like [0,3,1,...,4]
        outputs: 2-D array, first dim is batch size, second size 
            is number of classes.

    Returns:
        ap: ap value of batch samples
        cnt: Count of batch samples
    """
    cnt = 0
    ap = 0.
    for lb, op in zip(labels, outputs):
        op_argsort = np.argsort(op)[::-1]
        lb_int = int(lb)
        ap += 1.0 / (1+list(op_argsort).index(lb_int))
        cnt += 1
    return ap, cnt

