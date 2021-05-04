import sys
from typing import Dict, List
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from model.encoder import *
from utils import nn_util, util
from utils.dataset import Example
from utils.vocab import PAD_ID, Vocab
import torch
import torch.nn as nn

from transformers import XLNetModel, XLNetConfig, BertConfig, BertModel


class SequentialEncoder(Encoder):
    def __init__(self, config):
        super().__init__()

        self.vocab = vocab  = Vocab.load(config['vocab_file'])
        self.src_word_embed = nn.Embedding(len(vocab.source_tokens), config['source_embedding_size'])
        self.config = config

        self.decoder_cell_init = nn.Linear(config['source_encoding_size'], config['decoder_hidden_size'])

        if self.config['transformer'] == 'none':
            dropout = config['dropout']
            self.lstm_encoder = nn.LSTM(input_size=self.src_word_embed.embedding_dim,
                                        hidden_size=config['source_encoding_size'] // 2, num_layers=config['num_layers'],
                                        batch_first=True, bidirectional=True, dropout=dropout)

            self.dropout = nn.Dropout(dropout)

        elif self.config['transformer'] == 'bert':
            self.vocab_size = len(self.vocab.source_tokens) + 1

            state_dict = torch.load('saved_checkpoints/bert_2604/bert_pretrained_epoch_23_batch_140000.pth')

            keys_to_delete = ["cls.predictions.bias", "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight",
                            "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight", "cls.predictions.decoder.bias",
                            "cls.seq_relationship.weight", "cls.seq_relationship.bias"]

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['model'].items():
                if k in keys_to_delete: continue
                name = k[5:] # remove `bert.`
                new_state_dict[name] = v

            bert_config = BertConfig(vocab_size=self.vocab_size, max_position_embeddings=512, num_hidden_layers=6, hidden_size=256, num_attention_heads=4)
            self.bert_model = BertModel(bert_config)
            self.bert_model.load_state_dict(new_state_dict)

        elif self.config['transformer'] == 'xlnet':
            self.vocab_size = len(self.vocab.source_tokens) + 1

            state_dict = torch.load('saved_checkpoints/xlnet_2704/xlnet1_pretrained_epoch_13_iter_500000.pth')

            keys_to_delete = ["lm_loss.weight", "lm_loss.bias"]

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict['model'].items():
                if k in keys_to_delete: continue
                if k[:12] == 'transformer.': name = k[12:]
                else:                       name = k
                new_state_dict[name] = v

            xlnet_config = XLNetConfig(vocab_size=self.vocab_size, d_model=256, n_layer=12)
            self.xlnet_model = XLNetModel(xlnet_config)
            self.xlnet_model.load_state_dict(new_state_dict)
        else:
            print("Error! Unknown transformer type '{}'".format(self.config['transformer']))

    @property
    def device(self):
        return self.src_word_embed.weight.device

    @classmethod
    def default_params(cls):
        return {
            'source_encoding_size': 256,
            'decoder_hidden_size': 128,
            'source_embedding_size': 128,
            'vocab_file': None,
            'num_layers': 1
        }

    @classmethod
    def build(cls, config):
        params = util.update(SequentialEncoder.default_params(), config)

        return cls(params)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]):
        if self.config['transformer'] == 'bert':
            code_token_encoding, code_token_mask = self.encode_bert(tensor_dict['src_code_tokens'])
        elif self.config['transformer'] == 'xlnet':
            code_token_encoding, code_token_mask = self.encode_xlnet(tensor_dict['src_code_tokens'])
        elif self.config['transformer'] == 'none':
            code_token_encoding, code_token_mask, (last_states, last_cells) = self.encode_sequence(tensor_dict['src_code_tokens'])
        else:
            print("Error! Unknown transformer type '{}'".format(self.config['transformer']))
        # (batch_size, max_variable_mention_num)
        # variable_mention_positions = tensor_dict['variable_position']
        variable_mention_mask = tensor_dict['variable_mention_mask']
        variable_mention_to_variable_id = tensor_dict['variable_mention_to_variable_id']

        # (batch_size, max_variable_num)
        variable_encoding_mask = tensor_dict['variable_encoding_mask']
        variable_mention_num = tensor_dict['variable_mention_num']

        # # (batch_size, max_variable_mention_num, encoding_size)
        # variable_mention_encoding = torch.gather(code_token_encoding, 1, variable_mention_positions.unsqueeze(-1).expand(-1, -1, code_token_encoding.size(-1))) * variable_mention_positions_mask
        max_time_step = variable_mention_to_variable_id.size(1)
        variable_num = variable_mention_num.size(1)
        encoding_size = code_token_encoding.size(-1)

        variable_mention_encoding = code_token_encoding * variable_mention_mask.unsqueeze(-1)
        variable_encoding = torch.zeros(tensor_dict['batch_size'], variable_num, encoding_size, device=self.device)
        variable_encoding.scatter_add_(1,
                                       variable_mention_to_variable_id.unsqueeze(-1).expand(-1, -1, encoding_size),
                                       variable_mention_encoding) * variable_encoding_mask.unsqueeze(-1)
        variable_encoding = variable_encoding / (variable_mention_num + (1. - variable_encoding_mask) * nn_util.SMALL_NUMBER).unsqueeze(-1)

        if self.config['transformer'] == 'bert' or self.config['transformer'] == 'xlnet':
            context_encoding = dict(
                variable_encoding=variable_encoding,
                code_token_encoding=code_token_encoding,
                code_token_mask=code_token_mask
            )
        else:
            context_encoding = dict(
                variable_encoding=variable_encoding,
                code_token_encoding=code_token_encoding,
                code_token_mask=code_token_mask,
                last_states=last_states,
                last_cells=last_cells
            )

        context_encoding.update(tensor_dict)

        return context_encoding

    def encode_xlnet(self, input_ids):

        attention_mask = torch.ones_like(input_ids).float()
        attention_mask[input_ids == PAD_ID] = 0.0

        assert torch.max(input_ids) < self.vocab_size
        assert torch.min(input_ids) >= 0

        if torch.cuda.is_available():
            input_ids       = input_ids.cuda()
            attention_mask  = attention_mask.cuda()

        outputs = self.xlnet_model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs[0], attention_mask

    def encode_bert(self, input_ids):

        attention_mask = torch.ones_like(input_ids).float()
        attention_mask[input_ids == PAD_ID] = 0.0

        assert torch.max(input_ids) < self.vocab_size
        assert torch.min(input_ids) >= 0

        if torch.cuda.is_available():
            input_ids       = input_ids.cuda()
            attention_mask  = attention_mask.cuda()

        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        return outputs[0], attention_mask

    def encode_sequence(self, code_sequence):
        # (batch_size, max_code_length)
        # code_sequence = tensor_dict['src_code_tokens']

        # (batch_size, max_code_length, embed_size)
        code_token_embedding = self.src_word_embed(code_sequence)

        # (batch_size, max_code_length)
        code_token_mask = torch.ne(code_sequence, PAD_ID).float()
        # (batch_size)
        code_sequence_length = code_token_mask.sum(dim=-1).long()

        sorted_seqs, sorted_seq_lens, restoration_indices, sorting_indices = nn_util.sort_batch_by_length(code_token_embedding,
                                                                                                          code_sequence_length)

        packed_question_embedding = pack_padded_sequence(sorted_seqs, sorted_seq_lens.data.tolist(), batch_first=True)

        sorted_encodings, (last_states, last_cells) = self.lstm_encoder(packed_question_embedding)
        sorted_encodings, _ = pad_packed_sequence(sorted_encodings, batch_first=True)

        # apply dropout to the last layer
        # (batch_size, seq_len, hidden_size * 2)
        sorted_encodings = self.dropout(sorted_encodings)

        # (batch_size, question_len, hidden_size * 2)
        restored_encodings = sorted_encodings.index_select(dim=0, index=restoration_indices)

        # (num_layers, direction_num, batch_size, hidden_size)
        last_states = last_states.view(self.lstm_encoder.num_layers, 2, -1, self.lstm_encoder.hidden_size)
        last_states = last_states.index_select(dim=2, index=restoration_indices)
        last_cells = last_cells.view(self.lstm_encoder.num_layers, 2, -1, self.lstm_encoder.hidden_size)
        last_cells = last_cells.index_select(dim=2, index=restoration_indices)

        return restored_encodings, code_token_mask, (last_states, last_cells)

    @classmethod
    def to_tensor_dict(cls, examples: List[Example], next_examples=None, flips=None) -> Dict[str, torch.Tensor]:
        if next_examples is not None:
            max_time_step = max(e.source_seq_length + n.source_seq_length for e,n in zip(examples, next_examples))
        else:
            max_time_step = max(e.source_seq_length for e in examples)

        input = np.zeros((len(examples), max_time_step), dtype=np.int64)

        if next_examples is not None:
            seq_mask = torch.zeros((len(examples), max_time_step), dtype=torch.long)
        else:
            seq_mask = None

        variable_mention_to_variable_id = torch.zeros(len(examples), max_time_step, dtype=torch.long)
        variable_mention_mask = torch.zeros(len(examples), max_time_step)
        variable_mention_num = torch.zeros(len(examples), max(len(e.ast.variables) for e in examples))
        variable_encoding_mask = torch.zeros(variable_mention_num.size())

        for e_id, example in enumerate(examples):
            sub_tokens = example.sub_tokens
            input[e_id, :len(sub_tokens)] = example.sub_token_ids

            if next_examples is not None:
                next_example = next_examples[e_id]
                next_tokens = next_example.sub_tokens
                input[e_id, len(sub_tokens):len(sub_tokens)+len(next_tokens)] = next_example.sub_token_ids
                seq_mask[e_id, len(sub_tokens):] = 1
                # seq_mask[e_id, len(sub_tokens):len(sub_tokens)+len(next_tokens)] = 1

            variable_position_map = dict()
            var_name_to_id = {name: i for i, name in enumerate(example.ast.variables)}
            for i, sub_token in enumerate(sub_tokens):
                if sub_token.startswith('@@') and sub_token.endswith('@@'):
                    old_var_name = sub_token[2: -2]
                    if old_var_name in var_name_to_id:  # sometimes there are strings like `@@@@`
                        var_id = var_name_to_id[old_var_name]

                        variable_mention_to_variable_id[e_id, i] = var_id
                        variable_mention_mask[e_id, i] = 1.
                        variable_position_map.setdefault(old_var_name, []).append(i)

            for var_id, var_name in enumerate(example.ast.variables):
                try:
                    var_pos = variable_position_map[var_name]
                    variable_mention_num[e_id, var_id] = len(var_pos)
                except KeyError:
                    variable_mention_num[e_id, var_id] = 1
                    print(example.binary_file, f'variable [{var_name}] not found', file=sys.stderr)

            variable_encoding_mask[e_id, :len(example.ast.variables)] = 1.

        batch_dict =  dict(src_code_tokens=torch.from_numpy(input),
                            variable_mention_to_variable_id=variable_mention_to_variable_id,
                            variable_mention_mask=variable_mention_mask,
                            variable_mention_num=variable_mention_num,
                            variable_encoding_mask=variable_encoding_mask,
                            batch_size=len(examples))

        if next_examples is not None:
            batch_dict['next_seq_mask'] = seq_mask,
            batch_dict['next_sentence_label'] = torch.LongTensor(flips)

        return batch_dict

    def get_decoder_init_state(self, context_encoder, config=None):
        if 'last_cells' not in context_encoder:
            if self.config['init_decoder']:
                dec_init_cell = self.decoder_cell_init(torch.mean(context_encoder['code_token_encoding'], dim=1))
                dec_init_state = torch.tanh(dec_init_cell)
            else:
                dec_init_cell = dec_init_state = None

        elif 'last_cells' in context_encoder:
            fwd_last_layer_cell = context_encoder['last_cells'][-1, 0]
            bak_last_layer_cell = context_encoder['last_cells'][-1, 1]

            dec_init_cell = self.decoder_cell_init(torch.cat([fwd_last_layer_cell, bak_last_layer_cell], dim=-1))
            dec_init_state = torch.tanh(dec_init_cell)

        return dec_init_state, dec_init_cell

    def get_attention_memory(self, context_encoding, att_target='terminal_nodes'):
        assert att_target == 'terminal_nodes'

        memory = context_encoding['code_token_encoding']
        mask = context_encoding['code_token_mask']

        return memory, mask
