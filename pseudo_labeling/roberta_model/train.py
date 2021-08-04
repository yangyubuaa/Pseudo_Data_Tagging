"""训练器

每次迭代训练都需要重新构建验证集

"""
import numpy as np
import sys
sys.path.append("../")
import torch

from tqdm import tqdm

from pseudo_labeling.roberta_model.model import RobertaClsModel
from pseudo_labeling.data.dataset import train_eval_split

from torch.utils.data import DataLoader
from torch.optim import AdamW

from torch.nn.functional import cross_entropy

import logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config, iterations):
        self.iterations = iterations
        self.config = config
        logger.info("初始化trainer...")
        logger.info("加载模型...")
        self.train_model = RobertaClsModel(config)
        logger.info("加载数据...")
        self.train_set, self.eval_set = train_eval_split(config, iterations)
        logger.info("初始化Trainer完成...")

    def train(self):
        logger.info("执行训练...")
        logger.info("训练参数...")
        # 指定可用GPU数量
        device_ids = [0, 1, 2, 3]
        model = self.train_model

        if self.config["use_cuda"] and torch.cuda.is_available():
            # model = torch.nn.DataParallel(model)
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            # model = model.cuda()
            model = model.cuda(device=device_ids[0])
            # model = torch.nn.parallel.DistributedDataParallel(model)

        train_dataloader = DataLoader(dataset=self.train_set, batch_size=self.config["batch_size"], shuffle=True)
        eval_dataloader = DataLoader(dataset=self.eval_set, batch_size=self.config["batch_size"], shuffle=True)

        optimizer = AdamW(model.parameters(), self.config["LR"])

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num train examples = %d", len(self.train_set))
        logger.info("  Num eval examples = %d", len(self.eval_set))
        # logger.info("  Num test examples = %d", len(test_dataloader)*config["batch_size"])
        logger.info("  Num Epochs = %d", self.config["EPOCH"])
        logger.info("  Learning rate = %d", self.config["LR"])

        model.train()

        for epoch in range(self.config["EPOCH"]):
            for index, batch in tqdm(enumerate(train_dataloader)):
                optimizer.zero_grad()
                input_ids, attention_mask, token_type_ids = \
                    batch[0]["input_ids"].squeeze(), batch[0]["attention_mask"].squeeze(), batch[0][
                        "token_type_ids"].squeeze()
                label = batch[1]
                if self.config["use_cuda"] and torch.cuda.is_available():
                    input_ids, attention_mask, token_type_ids = \
                        input_ids.cuda(device=device_ids[0]), attention_mask.cuda(
                            device=device_ids[0]), token_type_ids.cuda(device=device_ids[0])
                    label = label.cuda(device=device_ids[0])
                model_output = model(input_ids, attention_mask, token_type_ids)
                train_loss = cross_entropy(model_output, label)
                train_loss.backward()
                print(train_loss)
                # print(model_output)
                # print(label)
                optimizer.step()
                if index > 0 and index % 20 == 0:
                    self.eval(model, eval_dataloader, device_ids)
        return self.train_model

    def eval(self, model, eval_dataloader, device_ids):
        # test
        model = model.eval()
        logger.info("eval!")
        loss_sum = 0

        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        # 创建混淆矩阵
        cls_nums = len(self.config["categories"])
        confuse_matrix = np.zeros((cls_nums, cls_nums))

        for index, batch in enumerate(eval_dataloader):
            if index % 5 == 0:
                print("eval{}/{}".format(str(index), str(len(eval_dataloader))))
            input_ids, attention_mask, token_type_ids = \
                batch[0]["input_ids"].squeeze(), batch[0]["attention_mask"].squeeze(), batch[0][
                    "token_type_ids"].squeeze()
            label = batch[1]
            if self.config["use_cuda"] and torch.cuda.is_available():
                input_ids, attention_mask, token_type_ids = \
                    input_ids.cuda(device=device_ids[0]), attention_mask.cuda(
                        device=device_ids[0]), token_type_ids.cuda(device=device_ids[0])
                label = label.cuda(device=device_ids[0])
            model_output = model(input_ids, attention_mask, token_type_ids)
            eval_loss = cross_entropy(model_output, label)
            loss_sum = loss_sum + eval_loss.item()

            pred = torch.argmax(model_output, dim=1)

            # print(model_output, pred)
            correct += (pred == label).sum().float()
            total += len(label)
            for index in range(len(pred)):
                confuse_matrix[label[index]][pred[index]] = confuse_matrix[label[index]][pred[index]] + 1

        logger.info("eval loss: {}".format(str(loss_sum / (len(eval_dataloader)))))
        logger.info("eval accu: {}".format(str((correct / total).cpu().detach().data.numpy())))
        logger.info("confuse_matrix:")
        for i in range(cls_nums):
            strs = self.config["categories"][i]
            for j in range(cls_nums):
                strs = strs + str(confuse_matrix[i][j]) + " |"
            logger.info(strs)

        for i in range(cls_nums):
            strs = self.config["categories"][i]
            p, r = 0, 0
            for j in range(cls_nums):
                p = p + confuse_matrix[j][i]
                r = r + confuse_matrix[i][j]
            strs = strs + " 精度 {}".format(str(confuse_matrix[i][i] / p)) + " 召回率 {}".format(
                str(confuse_matrix[i][i] / r))
            logger.info(strs)
