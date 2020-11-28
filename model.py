import torch
import torch.nn as nn


class ExtractiveModel(nn.Module):
    def __init__(self, bert_model, pos_cnt, media_cnt, embed_dim, use_bert_sum_words=True, use_pos=True,
                 use_media=True, dropout=0.1, num_classes=4, dim_feedforward=1024, simple_model=True):
        super(ExtractiveModel, self).__init__()
        self.bert = bert_model
        self.pos_embed = nn.Embedding(pos_cnt, embed_dim)
        self.media_embed = nn.Embedding(media_cnt, embed_dim)
        self.use_bert_sum_words = use_bert_sum_words
        self.use_media = use_media
        self.use_pos = use_pos

        self.simple_model = simple_model
        if simple_model:
            self.linear = nn.Linear(embed_dim, num_classes)
        else:
            self.linear1 = nn.Linear(embed_dim, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, num_classes)

            self.norm1 = nn.LayerNorm(embed_dim)
            # self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            # self.dropout2 = nn.Dropout(dropout)

            self.activation = nn.ReLU()

    def forward(self, input_ids, pos_ids, media_ids):
        # input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        # input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        # token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
        # model, vocab = get_pytorch_kobert_model()
        #
        input_mask = (input_ids != 0).type(torch.long)
        sequence_output, pooled_output = self.bert(input_ids, input_mask)
        if self.use_bert_sum_words:
            sentence_embed = torch.sum(sequence_output, dim=1)
        else:
            sentence_embed = pooled_output

        if self.use_pos:
            sentence_embed += self.pos_embed(pos_ids)

        if self.use_media:
            sentence_embed += self.media_embed(media_ids)

        if self.simple_model:
            logits = self.linear(sentence_embed)
        else:
            sentence_embed = self.dropout1(sentence_embed)
            sentence_embed = self.norm1(sentence_embed)
            # if hasattr(self, "activation"):
            logits = self.linear2(self.dropout(self.activation(self.linear1(sentence_embed))))
            # else:  # for backward compatibility
            #     src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return logits
