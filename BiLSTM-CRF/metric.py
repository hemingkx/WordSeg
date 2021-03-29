import os
import config
import logging


def calculate(x, y, res=None):
    """
    Gets entities from sequence.
    Args:
        x (list): sequence of words.
        y (list): sequence of labels.
        res: list of results
    Returns:
        res: list of entities.
    """
    if res is None:
        res = []
    entity = []
    prev_tag = 'O'  # start tag
    for i, tag in enumerate(y + ['P']):  # end tag
        if end_of_chunk(prev_tag, tag):
            res.append(entity)
            entity = []
        if start_of_chunk(prev_tag, tag) and tag != 'P':
            entity = [x[i]]
        elif tag != 'P':
            entity.append(x[i])
        else:
            continue
        prev_tag = tag
    return res


def end_of_chunk(prev_tag, tag):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    if prev_tag == 'E':
        chunk_end = True
    if tag == 'P':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'M' and tag == 'B':
        chunk_end = True
    if prev_tag == 'M' and tag == 'S':
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'O':
        chunk_start = True
    if prev_tag == 'S':
        chunk_start = True
    if prev_tag == 'E' and tag == 'M':
        chunk_start = True
    if prev_tag == 'E' and tag == 'E':
        chunk_start = True

    return chunk_start


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
