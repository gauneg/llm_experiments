import numpy as np


def check_and_insert(targ_dict: dict, gold_set: set, pred_set: set) -> dict:
    TP_INDEX = 0
    FP_INDEX = 1
    FN_INDEX = 2
    GOLD_CNT = 3
    PRED_CNT = 4
    TP = pred_set.intersection(gold_set)
    FP = pred_set - gold_set
    FN = gold_set - pred_set
    for elem in TP:
        if elem not in targ_dict.keys():
            targ_dict[elem] = [0, 0, 0, 0, 0]
        targ_dict[elem][TP_INDEX] += 1

    for elem in FP:
        if elem not in targ_dict.keys():
            targ_dict[elem] = [0, 0, 0, 0, 0]
        targ_dict[elem][FP_INDEX] += 1

    for elem in FN:
        if elem not in targ_dict.keys():
            targ_dict[elem] = [0, 0, 0, 0, 0]
        targ_dict[elem][FN_INDEX] += 1

    for elem in gold_set:
        if elem not in targ_dict.keys():
            targ_dict[elem] = [0, 0, 0, 0, 0]
        targ_dict[elem][GOLD_CNT] += 1

    for elem in pred_set:
        if elem not in targ_dict.keys():
            targ_dict[elem] = [0, 0, 0, 0, 0]
        targ_dict[elem][PRED_CNT] += 1

    return targ_dict


def calc_labs(pred_gold_lab_list):
    result_count_dict = {}
    accumulated_score = [0, 0, 0]  # PRECISION, RECALL, F1
    for i, (pred, gold) in enumerate(pred_gold_lab_list):
        pred_set = set(pred)
        gold_set = set(gold)
        result_count_dict = check_and_insert(result_count_dict, gold_set, pred_set)

    TP, FP, FN, T_LABS, T_PRED = np.array(list(result_count_dict.values())).sum(axis=0)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2 * precision * recall / (precision+recall)
    return precision, recall, f1