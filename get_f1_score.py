import numpy as np

def calc_p2p(predict, actual):
    tp = np.sum(predict * actual)
    tn = np.sum((1 - predict) * (1 - actual))
    fp = np.sum(predict * (1 - actual))
    fn = np.sum((1 - predict) * actual)
    precision = (tp + 0.000001) / (tp + fp + 0.000001)
    recall = (tp + 0.000001) / (tp + fn + 0.000001)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall, tp, tn, fp, fn

def point_adjust(score, label, thres):
    predict = score >= thres
    actual = label > 0.1
    anomaly_state = False
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    return predict, actual

def best_f1_without_pointadjust(score, label):
    max_th = np.percentile(score, 99.91)
    min_th = float(score.min())
    grain = 2000
    max_f1_1 = 0.0
    max_f1_th_1 = 0.0
    max_pre = 0.0
    max_recall = 0.0
    for i in range(0, grain + 3):
        thres = (max_th - min_th) / grain * i + min_th
        actual = label
        predict = score >= thres
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1_1:
            max_f1_1 = f1
            max_f1_th_1 = thres
            max_pre = precision
            max_recall = recall
    predict, actual = point_adjust(score, label, max_f1_th_1)
    return max_f1_1, max_pre,max_recall,predict


def best_f1(score, label):
    max_th = np.percentile(score, 99.91)
    min_th = float(score.min())
    grain = 2000
    max_f1 = 0.0
    max_f1_th = 0.0
    pre = 0.0
    rec = 0.0
    for i in range(0, grain + 3):
        thres = (max_th - min_th) / grain * i + min_th
        predict, actual = point_adjust(score, label, thres=thres)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_th = thres
            pre = precision
            rec = recall
    predict, actual = point_adjust(score, label, max_f1_th)
    return max_f1,pre,rec, predict

def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0
    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0
    return new_predict

def delay_f1(score, label, k=7):
    max_th = np.percentile(score, 99.91)
    min_th = float(score.min())
    grain = 2000
    max_f1 = 0.0
    max_f1_th = 0.0
    pre = 0.0
    rec = 0.0
    for i in range(0, grain + 3):
        thres = (max_th - min_th) / grain * i + min_th
        predict = score >= thres
        predict= get_range_proba(predict,label,k)
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, label)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_th = thres
            pre = precision
            rec = recall
        predict= get_range_proba(score>=max_f1_th,label,k)
    return max_f1,pre,rec,predict