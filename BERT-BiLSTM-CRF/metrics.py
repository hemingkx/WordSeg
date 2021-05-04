import os
import config
import logging


def get_entities(seq):
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):
            chunks.append((begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag

    return chunks


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
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'M' and tag == 'B':
        chunk_end = True
    if prev_tag == 'M' and tag == 'S':
        chunk_end = True
    if prev_tag == 'M' and tag == 'O':
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

    if prev_tag == 'O' and tag == 'M':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'M':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'M':
        chunk_start = True
    if prev_tag == 'E' and tag == 'E':
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred):
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

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


def calculate(x, y):
    """
    Gets words of entities from sequence.
    Args:
        x (list): sequence of words.
        y (list): sequence of labels.
    Returns:
        res: list of entities.
    """
    res = []
    entity = []
    prev_tag = 'O'  # start tag
    for i, tag in enumerate(y + ['O']):  # end tag
        if end_of_chunk(prev_tag, tag):
            res.append(entity)
            entity = []
        if start_of_chunk(prev_tag, tag) and i < len(x):
            entity = [x[i]]
        elif i < len(x):
            entity.append(x[i])
        else:
            continue
        prev_tag = tag
    return res


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
    score = f1_score(preds, tags)
    print("f1 score: {}".format(score))
    output_write(sents, preds)


def output2res():
    """write results into output.txt for f1 calculation"""
    words_list = []
    with open(config.output_dir, 'r', encoding='utf-8') as f:
        inline = False  # 上一句是否还未结束
        one_line = []
        for line in f:
            if line[-4] == '@':
                one_line.extend(line[:-4])
                inline = True
            else:
                if inline:
                    one_line.extend(line)
                    words_list.append("".join(one_line))
                    one_line = []
                    inline = False
                else:
                    words_list.append(line)
    with open(config.res_dir, "w") as f:
        for line in words_list:
            f.write(line)


if __name__ == "__main__":
    # f1_test()
    output2res()

