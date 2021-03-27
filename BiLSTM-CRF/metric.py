import os
import config
import logging


def calculate(x, y, res=None):
    if res is None:
        res = []
    entity = []
    for j in range(len(x)):
        if y[j] == 'B':
            entity = [x[j]]
        elif y[j] == 'M' and len(entity) != 0:
            entity.append(x[j])
        elif y[j] == 'E' and len(entity) != 0:
            entity.append(x[j])
            res.append(entity)
            entity = []
        elif y[j] == 'S':
            entity = [x[j]]
            res.append(entity)
            entity = []
        else:
            entity = []
    return res


def f1_score(sents, preds, tags):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        sents: 2d array, sentence list
        tags : 2d array. Ground truth (correct) target values.
        preds : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        tags = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(sentence, pred, tags)
        0.50
    """
    entityres = []
    entityall = []
    for idx, (t, p) in enumerate(zip(tags, preds)):
        entityres = calculate(sents[idx], p, entityres)
        entityall = calculate(sents[idx], t, entityall)

    # print("pred:", entityres)
    # print("labels:", entityall)
    nb_correct = len([i for i in entityres if i in entityall])
    nb_pred = len(entityres)
    nb_true = len(entityall)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return score, p, r


def bad_case(sents, preds, tags):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, 'w')
    for idx, (t, p) in enumerate(zip(tags, preds)):
        if t == p:
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(sents[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")


def output_write(sents, preds):
    """write results into output.txt for f1 calculation"""
    with open(config.output_dir, "w") as f:
        for (s, p) in zip(sents, preds):
            res = calculate(s, p)
            for entity in res:
                for w in entity:
                    f.write(w)
                f.write('  ')
            f.write("\n")


def f1_test():
    sents = [['机', '器', '人', '迎', '客', '小', '姐', '（', '图', '片', '）'], ['降', '水', '概', '率', '2', '0', '％']]
    tags = [['B', 'M', 'E', 'S', 'S', 'B', 'E', 'S', 'B', 'E', 'S'], ['B', 'E', 'B', 'E', 'B', 'M', 'E']]
    preds = [['B', 'E', 'S', 'B', 'E', 'B', 'E', 'S', 'B', 'E', 'S'], ['S', 'S', 'S', 'S', 'B', 'M', 'E']]
    score, p, r = f1_score(sents, preds, tags)
    print("f1 score: {}, precision:{}, recall: {}".format(score, p, r))
    output_write(sents, preds)


if __name__ == "__main__":
    f1_test()
