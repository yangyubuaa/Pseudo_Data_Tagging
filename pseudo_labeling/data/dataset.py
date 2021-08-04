import os
import torch
import random
import yaml
from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import AutoTokenizer

import logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainAndEvalDataset(Dataset):
    def __init__(self, tokenized_list, label_list):
        self.dataset = tokenized_list
        self.labels = label_list

    def __getitem__(self, item):
        return self.dataset[item], torch.tensor(int(self.labels[item]))

    def __len__(self):
        return len(self.dataset)


def train_eval_split(config, iterations):
    """划分训练集和验证集，并且进行tokenizer后封装并返回

    :return: torch.utils.data.Dataset子类
    """
    roberta_path = os.path.join(config["root"], config["roberta_model_path"])
    print(roberta_path)
    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    logger.info("数据读取...当前迭代轮次：{}".format(str(iterations)))
    datas, unlabeded_data_nums = load_train_file(config, iterations)
    random.shuffle(datas)
    train_rate = config["train_rate"]
    train_nums = int(len(datas) * train_rate)
    train_set = datas[:train_nums]
    eval_set = datas[train_nums:]

    logger.info("数据集{}条数据".format(str(len(datas))))
    logger.info("其中，训练集{}条数据".format(str(len(train_set))))
    logger.info("其中，验证集{}条数据".format(str(len(eval_set))))
    logger.info("未标记{}条数据".format(str(unlabeded_data_nums)))

    train_set_tokenized = list()
    train_set_label = list()

    eval_set_tokenized = list()
    eval_set_label = list()
    logger.info("tokenizer训练集")
    for i in tqdm(train_set):
        i_split = i.split("\t")
        tokenized = tokenizer(i_split[0], return_tensors="pt", max_length=20, padding="max_length", truncation=True)
        train_set_tokenized.append(tokenized)
        train_set_label.append(i_split[1])

    logger.info("tokenizer验证集")
    for j in tqdm(eval_set):
        j_split = j.split("\t")
        tokenized = tokenizer(j_split[0], return_tensors="pt", max_length=20, padding="max_length", truncation=True)
        eval_set_tokenized.append(tokenized)
        eval_set_label.append(j_split[1])

    return TrainAndEvalDataset(train_set_tokenized, train_set_label), TrainAndEvalDataset(eval_set_tokenized, eval_set_label)


def load_train_file(config, iterations):
    """加载训练文件

    :param iterations:
    :return:
    """
    logger.info("当前读取：{}".format(config["label_file_name"]))
    label_data = load_one_pseudo_file(config["root"] + "/" + config["label_file_name"])
    pseudo_datas = list()

    pseudo_file_name = config["pseudo_file_name"]
    if iterations > 0:
        for i in range(1, iterations + 1):
            logger.info("当前读取：{}".format(pseudo_file_name+str(i)+".txt"))
            pseudo_datas.append(load_one_pseudo_file(config["root"] + config["pseudo_file_path"] + "/" + pseudo_file_name+str(i)+".txt"))

    datas = list()
    for i in label_data:
        datas.append(i)

    for p in pseudo_datas:
        for j in p:
            datas.append(j)
    with open(os.path.join(config["root"], config["unlabel_file_name"]), "r", encoding="utf-8") as r:
        unlabeled_data_nums = len(r.readlines())

    return datas, unlabeled_data_nums


def load_one_pseudo_file(file_name):
    with open(file_name, "r", encoding="utf-8") as pseudo_r:
        data = [line.strip() for line in pseudo_r.readlines()]
    return data


if __name__ == '__main__':
    with open(r'../../config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    train, eval = train_eval_split(config, 3)