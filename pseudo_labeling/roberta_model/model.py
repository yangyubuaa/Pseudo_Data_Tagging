import os
import torch
from transformers import RobertaTokenizer, RobertaModel


class RobertaClsModel(torch.nn.Module):
    def __init__(self, config):
        super(RobertaClsModel, self).__init__()
        self.config = config
        self.roberta_path = os.path.join(config["root"], config["roberta_model_path"])
        self.roberta_model = RobertaModel.from_pretrained(self.roberta_path)
        self.dense = torch.nn.Linear(1024, 2)
    def forward(self, input_ids, attention_mask, token_type_ids):
        roberta_features = self.roberta_model(input_ids, attention_mask, token_type_ids)
        last_hidden_state = roberta_features.last_hidden_state
        features = last_hidden_state[:, 0, :].squeeze()
        # print(features.shape)
        logit = self.dense(features)
        return logit
