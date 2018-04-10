from nose.tools import *
from utils.metric import evaluate, AP, mAP
import numpy as np

labels = ['nnynn',
 'nnnny',
 'nnnny',
 'ynnnn',
 'nnnny',
 'ynnnn',
 'nnymn',
 'nnnyn',
 'nnnyn',
 'nmynn']

preds = np.array([[0.0057, 0.3725, 0.547, 0.0229, 0.0519],
 [0.0444, 0.3707, 0.1504, 0.0442, 0.3903],
 [0.1077, 0.5922, 0.0348, 0.0145, 0.2508],
 [0.077, 0.4272, 0.2941, 0.198, 0.0037],
 [0.0154, 0.1047, 0.0622, 0.0659, 0.7517],
 [0.0134, 0.0169, 0.6489, 0.3102, 0.0105],
 [0.1991, 0.5514, 0.2246, 0.0169, 0.008],
 [0.2832, 0.2654, 0.0475, 0.3944, 0.0095],
 [0.2131, 0.4011, 0.0753, 0.2114, 0.0991],
 [0.0445, 0.7361, 0.1946, 0.0208, 0.004]])

def metric_tests():
    # Test evaluate
    assert_almost_equal(evaluate(preds, labels), 0.4444444)

    # Test AP
    assert_almost_equal(AP(preds, labels), 0.4959444)

    # Test mAP
    preds_list = [preds, preds]
    labels_list = [labels, labels]
    assert_almost_equal(mAP(preds_list, labels_list), 0.4959444)

