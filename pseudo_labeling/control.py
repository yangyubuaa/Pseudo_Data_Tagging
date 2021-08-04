"""伪标签方式流程控制

"""
import os
import yaml
import torch

from tqdm import tqdm
from transformers import AutoTokenizer

from roberta_model.train import Trainer
import logging
logging.basicConfig(level = logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PseudoLabelMethod:
    def __init__(self, config):
        self.config = config
        self.max_iterations = config["max_iterations"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["root"] + config["roberta_model_path"])
        self.device_ids = [0, 1, 2, 3]

    def exec(self):
        logger.debug("执行维伪标签算法...")
        logger.debug("最大迭代次数：{}".format(str(self.max_iterations)))

        for i in range(1, self.max_iterations):
            logger.debug("*********************************************")
            logger.debug("当前迭代次数：{}".format(str(i)))
            logger.debug("构建训练器...")
            trainer = Trainer(self.config, i)
            trained_model = trainer.train()
            logger.debug("训练完成...加载unlabeled data进行数据标注...")
            self.tagging(trained_model, i)

    def tagging(self, trained_model, i):
        """使用训练完的模型进行数据标注

        :param trained_model:
        :return:
        """
        unlabeled_data_path = os.path.join(self.config["root"], self.config["unlabel_file_name"])
        with open(unlabeled_data_path, "r", encoding="utf-8") as unlabel_read:
            unlabel_datas = [i.strip() for i in unlabel_read.readlines()]

        logger.info("当前迭代轮次：{}".format(str(i)))
        logger.info("未标注数据：{}条".format(str(len(unlabel_datas))))
        logger.info("进行数据标注...")

        tagged_list = list()
        untagged_list = list()
        index = 0
        for untagged_line in tqdm(unlabel_datas):
            if index % 1000 == 0:
                logger.info("标注成功:{};标注失败:{}条数据".format(str(len(tagged_list)), str(len(untagged_list))))
            index = index + 1
            logit = self.predict(trained_model, untagged_line)
            # print(logit)
            THRESHOLD = self.config["threshold"]
            if logit[0] > THRESHOLD:
                label = 0
                tagged_list.append(untagged_line + "\t" + str(label))
                continue
            if logit[1] > THRESHOLD:
                label = 1
                tagged_list.append(untagged_line + "\t" + str(label))
                continue
            untagged_list.append(untagged_line)
        logger.info("标注成功:{};标注失败:{}条数据".format(str(len(tagged_list)), str(len(untagged_list))))
        logger.info("更新新标注文件...")
        new_file_name = self.config["root"] + "/" + self.config["pseudo_file_path"] + "/" + self.config["pseudo_file_name"] + str(i+1) + ".txt"
        with open(new_file_name, "w", encoding="utf-8") as iter_new_file:
            for i in tagged_list:
                iter_new_file.write(i + "\n")
        logger.info("更新未标注文件...")
        unlabel_file_name = self.config["root"] + "/" + self.config["unlabel_file_name"]
        with open(unlabel_file_name, "w", encoding="utf-8") as iter_unlabel_file:
            for i in untagged_list:
                iter_unlabel_file.write(i + "\n")

    def predict(self, trained_model, data):
        # print(data)
        tokenized = self.tokenizer(data, return_tensors="pt", max_length=20, padding="max_length", truncation=True)
        if self.config["use_cuda"] and torch.cuda.is_available():
            input_ids, attention_mask, token_type_ids = \
                tokenized["input_ids"].cuda(device=self.device_ids[0]), tokenized["attention_mask"].cuda(
                    device=self.device_ids[0]), tokenized["token_type_ids"].cuda(device=self.device_ids[0])
        logit = trained_model(input_ids, attention_mask, token_type_ids)
        return logit.detach().cpu().numpy().tolist()


if __name__ == "__main__":
    with open(r'../config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    pseudu = PseudoLabelMethod(config)
    pseudu.exec()
    # with open(config["root"] + "///" + "dataset/tagged/label.txt", "r", encoding="utf-8") as r:
    #     print(r.readlines())