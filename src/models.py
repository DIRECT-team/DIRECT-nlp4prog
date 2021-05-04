from helper import *
from model.graph_encoder import GraphASTEncoder
from transformers import BertConfig, BertModel, XLNetConfig, XLNetModel
from utils.vocab import PAD_ID, Vocab

class RenamingModelDecoderOnly(nn.Module):

    def __init__(self, vocab, top_k, config, device):
        super(RenamingModelDecoderOnly, self).__init__()

        self.vocab      = vocab
        self.top_k      = top_k

        self.target_vocab_size = len(self.vocab.all_subtokens) + 1

        bert_config = BertConfig(vocab_size=self.target_vocab_size, max_position_embeddings=1000, num_hidden_layers=6, hidden_size=256, num_attention_heads=4, is_decoder=True)
        self.bert_decoder = BertModel(bert_config)

        state_dict = torch.load('saved_checkpoints/bert_0905/bert_decoder_epoch_19_batch_220000.pth', map_location=device)

        keys_to_delete = ["cls.predictions.bias", "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight",
                        "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight", "cls.predictions.decoder.bias",
                        "cls.seq_relationship.weight", "cls.seq_relationship.bias"]

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            if k in keys_to_delete: continue
            if 'crossattention' in k: continue
            name = k[5:] # remove `bert.`
            new_state_dict[name] = v

        for key in new_state_dict:
            self.bert_decoder.state_dict()[key].copy_(new_state_dict[key])

        self.fc_final = nn.Linear(256, self.target_vocab_size)
        self.fc_final.weight.data = state_dict['model']['cls.predictions.decoder.weight']

    def forward(self, src_tokens, target_tokens):
        assert torch.max(target_tokens) < self.target_vocab_size
        assert torch.min(target_tokens) >= 0

        decoder_attention_mask = torch.ones_like(target_tokens).float().to(target_tokens.device)
        decoder_attention_mask[target_tokens == PAD_ID] = 0.0

        decoder_output = self.bert_decoder(input_ids=target_tokens, attention_mask=decoder_attention_mask)[0]

        predictions    = self.fc_final(decoder_output)

        return predictions

    def predict(self, src_tokens, src_mask, approx=False):
        end_token   = self.vocab.all_subtokens.word2id['</s>']
        start_token = self.vocab.all_subtokens.word2id['<s>']
        batch_size  = src_tokens.shape[0]


        source_vocab_to_target  = {self.vocab.source_tokens.word2id[t] : self.vocab.all_subtokens.word2id[t] for t in self.vocab.source_tokens.word2id.keys()}
        src_target_maps         = []
        confidences             = []

        for i in range(batch_size):

            if src_tokens[i][0] != start_token:
                input_sequence      = torch.zeros(src_tokens.shape[1] + 1).to(src_tokens.device)
                input_mask          = torch.zeros(src_mask.shape[1] + 1).to(src_mask.device)
                input_sequence[1:]  = src_tokens[i]
                input_mask[1:]      = src_mask[i]
            else:
                input_sequence      = src_tokens[i]
                input_mask          = src_mask[i]

            num_vars        = int(input_mask.sum())
            seq_len         = torch.sum((input_sequence != PAD_ID).long())
            generated_seqs  = torch.zeros(1, min(seq_len + 10 * num_vars, 1000)).long().to(src_tokens.device)

            source_marker   = 0
            gen_markers     = torch.LongTensor ([0]).to(generated_seqs.device)
            prior_probs     = torch.FloatTensor([0]).to(generated_seqs.device)

            candidate_maps = [{}]

            for _ in range(num_vars):
                # Filling up the known (non-identifier) tokens
                while source_marker < seq_len and input_mask[source_marker] != 1:
                    token           = input_sequence[source_marker]
                    values          = source_vocab_to_target[token.item()] * torch.ones_like(gen_markers).to(generated_seqs.device)

                    generated_seqs  = torch.scatter(generated_seqs, 1, gen_markers.unsqueeze(1), values.unsqueeze(1))

                    source_marker   += 1
                    gen_markers     += 1

                if source_marker >= seq_len: break

                curr_var = input_sequence[source_marker].item()

                if curr_var in candidate_maps[0]:
                    # If we've seen this variable before, just use the previous predictions and update the scores
                    # Note - it's enough to check candidate_maps[0] because if it is in the first map, it is in all of them
                    if approx is True:
                        source_marker += 1
                        continue

                    orig_markers = gen_markers.clone()

                    for j in range(len(candidate_maps)):
                        pred = candidate_maps[j][curr_var]
                        generated_seqs[j][gen_markers[j] : gen_markers[j] + len(pred)] = torch.LongTensor(pred).to(generated_seqs.device)
                        gen_markers[j] += len(pred)

                    decoder_attention_mask = torch.ones_like(generated_seqs).float().to(generated_seqs.device)
                    decoder_attention_mask[generated_seqs == PAD_ID] = 0.0

                    decoder_output  = self.bert_decoder(input_ids=generated_seqs, attention_mask=decoder_attention_mask)[0]

                    probabilities   = F.log_softmax(self.fc_final(decoder_output), dim=-1)

                    # Add up the scores of the token at the __next__ time step

                    scores          = torch.zeros(generated_seqs.shape[0]).to(generated_seqs.device)
                    active          = torch.ones(generated_seqs.shape[0]).long().to(generated_seqs.device)
                    temp_markers    = orig_markers

                    while torch.sum(active) != 0:
                        position_probs  = torch.gather(probabilities, 1, (temp_markers - 1).reshape(-1, 1, 1).repeat(1, 1, probabilities.shape[2])).squeeze(1)
                        curr_tokens     = torch.gather(generated_seqs, 1, temp_markers.unsqueeze(1))
                        tok_probs       = torch.gather(position_probs, 1, curr_tokens).squeeze(1)

                        tok_probs       *= active
                        scores          += tok_probs

                        active          *= (temp_markers != (gen_markers-1)).long()
                        temp_markers    += active

                    # Update the prior probabilities
                    prior_probs         = prior_probs + scores

                else:
                    # You encounter a new variable which hasn't been seen before
                    # Generate <beam_width> possibilities for its name
                    generated_seqs, gen_markers, prior_probs, candidate_maps    = self.beam_search(generated_seqs, gen_markers, prior_probs, candidate_maps, curr_var,
                                                                                                   beam_width=5, top_k=self.top_k)

                source_marker += 1

            final_ind       = torch.argmax(prior_probs)
            confidence      = torch.max(prior_probs).item()
            src_target_map  = candidate_maps[final_ind]

            src_target_maps.append(src_target_map)
            confidences.append(confidence)

        return src_target_maps, confidences

    def beam_search(self, generated_seqs, gen_markers, prior_probs, candidate_maps, curr_var, beam_width=5, top_k=10):

        if generated_seqs.shape[0] * beam_width < top_k:
            beam_width = top_k

        active      = torch.ones_like(gen_markers).to(gen_markers.device)
        beam_alpha  = 0.7
        end_token   = self.vocab.all_subtokens.word2id['</s>']

        candidate_maps = candidate_maps
        orig_markers    = gen_markers.clone()

        for _ in range(10): # Predict at most 10 subtokens
            decoder_attention_mask = torch.ones_like(generated_seqs).float().to(generated_seqs.device)
            decoder_attention_mask[generated_seqs == PAD_ID] = 0.0

            decoder_output  = self.bert_decoder(input_ids=generated_seqs, attention_mask=decoder_attention_mask)[0]
            probabilities   = F.log_softmax(self.fc_final(decoder_output), dim=-1)
            # Gather the predictions at the current markers
            # (gen_marker - 1) because prediction happens one step ahead
            probabilities   = torch.gather(probabilities, 1, (gen_markers - 1).reshape(-1, 1, 1).repeat(1, 1, probabilities.shape[2])).squeeze(1)

            probs, preds    = probabilities.sort(dim=-1, descending=True)

            probs               *= active.unsqueeze(1)  # Set log prob of non-active ones to 0
            preds[active == 0]  = end_token  # Set preds of non-active ones to the end token (ie, remain unchanged)

            # Repeat active ones only once. Repeat the rest beam_width no. of times.
            filter_mask = torch.ones((preds.shape[0], beam_width)).long().to(preds.device)
            filter_mask *= active.unsqueeze(1)
            filter_mask[:, 0][active == 0] = 1
            filter_mask = filter_mask.reshape(-1)

            preds           = preds[:, :beam_width].reshape(-1)[filter_mask == 1]
            probs           = probs[:, :beam_width].reshape(-1)[filter_mask == 1]

            generated_seqs  = torch.repeat_interleave(generated_seqs,   beam_width, dim=0)[filter_mask == 1]
            orig_markers    = torch.repeat_interleave(orig_markers,     beam_width, dim=0)[filter_mask == 1]
            gen_markers     = torch.repeat_interleave(gen_markers,      beam_width, dim=0)[filter_mask == 1]
            active          = torch.repeat_interleave(active,           beam_width, dim=0)[filter_mask == 1]
            prior_probs     = torch.repeat_interleave(prior_probs,      beam_width, dim=0)[filter_mask == 1]

            candidate_maps = [item.copy() for item in candidate_maps for _ in range(beam_width)]
            candidate_maps = [candidate_maps[i] for i in range(len(candidate_maps)) if filter_mask[i] == 1]

            generated_seqs.scatter_(1, gen_markers.unsqueeze(1), preds.unsqueeze(1))

            # lengths       = (gen_markers - gen_marker + 1).float()
            # penalties     = torch.pow(5 + lengths, beam_alpha) / math.pow(6, beam_alpha)
            penalties       = torch.ones_like(probs).to(probs.device)

            updated_probs   = probs + prior_probs

            sort_inds       = (updated_probs / penalties).argsort(descending=True)
            updated_probs   = updated_probs[sort_inds]

            prior_probs     = updated_probs[:top_k]

            new_preds       = preds         [sort_inds[:top_k]]
            generated_seqs  = generated_seqs[sort_inds[:top_k]]
            gen_markers     = gen_markers   [sort_inds[:top_k]]
            active          = active        [sort_inds[:top_k]]
            orig_markers    = orig_markers  [sort_inds[:top_k]]

            candidate_maps = [candidate_maps[ind.item()] for ind in sort_inds[:top_k]]

            active          = active * (new_preds != end_token).long()
            gen_markers     += active

            if torch.sum(active) == 0: break

        # gen_markers are pointing at the end_token. Move them one ahead
        gen_markers += 1

        assert generated_seqs.shape[0] == top_k

        for i in range(top_k):
            candidate_maps[i][curr_var] = generated_seqs[i][orig_markers[i] : gen_markers[i]].cpu().tolist()

        return generated_seqs, gen_markers, prior_probs, candidate_maps


class RenamingModel(nn.Module):

    def __init__(self, vocab, top_k, config, device):
        super(RenamingModel, self).__init__()

        self.vocab = vocab
        self.top_k = top_k
        self.source_vocab_size = len(self.vocab.source_tokens) + 1

        state_dict = torch.load('saved_checkpoints/bert_encoder.pth', map_location=device)

        keys_to_delete = ["cls.predictions.bias", "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight",
                        "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight", "cls.predictions.decoder.bias",
                        "cls.seq_relationship.weight", "cls.seq_relationship.bias"]

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            if k in keys_to_delete: continue
            name = k[5:] # remove `bert.`
            new_state_dict[name] = v

        bert_config = BertConfig(vocab_size=self.source_vocab_size, max_position_embeddings=512, num_hidden_layers=6, hidden_size=256, num_attention_heads=4, output_attentions=True)
        self.bert_encoder = BertModel(bert_config)
        self.bert_encoder.load_state_dict(new_state_dict)

        self.target_vocab_size = len(self.vocab.all_subtokens) + 1

        bert_config = BertConfig(vocab_size=self.target_vocab_size, max_position_embeddings=1000, num_hidden_layers=6, hidden_size=256, num_attention_heads=4, is_decoder=True, output_attentions=True)
        self.bert_decoder = BertModel(bert_config)

        state_dict = torch.load('saved_checkpoints/bert_decoder.pth', map_location=device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            if k in keys_to_delete: continue
            if 'crossattention' in k: continue
            name = k[5:] # remove `bert.`
            new_state_dict[name] = v

        for key in new_state_dict:
            self.bert_decoder.state_dict()[key].copy_(new_state_dict[key])

        self.fc_final = nn.Linear(256, self.target_vocab_size)
        self.fc_final.weight.data = state_dict['model']['cls.predictions.decoder.weight']

    def forward(self, src_tokens, target_tokens):
        encoder_attention_mask = torch.ones_like(src_tokens).float().to(src_tokens.device)
        encoder_attention_mask[src_tokens == PAD_ID] = 0.0

        assert torch.max(src_tokens) < self.source_vocab_size
        assert torch.min(src_tokens) >= 0
        assert torch.max(target_tokens) < self.target_vocab_size
        assert torch.min(target_tokens) >= 0

        encoder_output = self.bert_encoder(input_ids=src_tokens, attention_mask=encoder_attention_mask)[0]

        decoder_attention_mask = torch.ones_like(target_tokens).float().to(target_tokens.device)
        decoder_attention_mask[target_tokens == PAD_ID] = 0.0

        decoder_output = self.bert_decoder(input_ids=target_tokens, attention_mask=decoder_attention_mask,
                                           encoder_hidden_states=encoder_output, encoder_attention_mask=encoder_attention_mask)[0]

        predictions    = self.fc_final(decoder_output)

        return predictions

    def predict(self, src_tokens, src_mask, approx=False, top5=True):
        end_token   = self.vocab.all_subtokens.word2id['</s>']
        start_token = self.vocab.all_subtokens.word2id['<s>']
        batch_size  = src_tokens.shape[0]

        encoder_attention_mask = torch.ones_like(src_tokens).float().to(src_tokens.device)
        encoder_attention_mask[src_tokens == PAD_ID] = 0.0

        assert torch.max(src_tokens) < self.source_vocab_size
        assert torch.min(src_tokens) >= 0

        encoder_output, _, enc_attentions = self.bert_encoder(input_ids=src_tokens, attention_mask=encoder_attention_mask)

        source_vocab_to_target  = {self.vocab.source_tokens.word2id[t] : self.vocab.all_subtokens.word2id[t] for t in self.vocab.source_tokens.word2id.keys()}
        batch_preds         = []
        batch_confs         = []

        for i in range(batch_size):

            if src_tokens[i][0] != start_token:
                input_sequence      = torch.zeros(src_tokens.shape[1] + 1).long().to(src_tokens.device)
                input_mask          = torch.zeros(src_mask.shape[1] + 1).to(src_mask.device)
                input_sequence[1:]  = src_tokens[i]
                input_mask[1:]      = src_mask[i]
            else:
                input_sequence      = src_tokens[i]
                input_mask          = src_mask[i]

            num_vars        = int(input_mask.sum())
            seq_len         = torch.sum((input_sequence != PAD_ID).long())
            generated_seqs  = torch.zeros(1, min(seq_len + 10 * num_vars, 1000)).long().to(src_tokens.device)

            # In the generated sequence, what placeholder variables do the generated names correspond to?
            # -1 -> non-variable portion of generated sequence
            gen_var_ids     = -1 * torch.ones(1, min(seq_len + 10 * num_vars, 1000)).long().to(src_tokens.device)

            source_marker   = 0
            gen_markers     = torch.LongTensor ([0]).to(generated_seqs.device)
            prior_probs     = torch.FloatTensor([0]).to(generated_seqs.device)

            candidate_maps = [{}]

            for _ in range(num_vars):
                # Filling up the known (non-identifier) tokens
                while source_marker < seq_len and input_mask[source_marker] != 1:
                    token           = input_sequence[source_marker]
                    values          = source_vocab_to_target[token.item()] * torch.ones_like(gen_markers).long().to(generated_seqs.device)

                    generated_seqs  = torch.scatter(generated_seqs, 1, gen_markers.unsqueeze(1), values.unsqueeze(1))

                    source_marker   += 1
                    gen_markers     += 1

                if source_marker >= seq_len: break

                curr_var = input_sequence[source_marker].item()

                if curr_var in candidate_maps[0]:
                    if approx is True: # Just predict at first occurrence of variable
                        source_marker += 1
                        continue
                    # If we've seen this variable before, just use the previous predictions and update the scores
                    # Note - it's enough to check candidate_maps[0] because if it is in the first map, it is in all of them

                    orig_markers = gen_markers.clone()

                    for j in range(len(candidate_maps)):
                        pred = candidate_maps[j][curr_var]
                        generated_seqs[j][gen_markers[j] : gen_markers[j] + len(pred)] = torch.LongTensor(pred).to(generated_seqs.device)
                        gen_var_ids   [j][gen_markers[j] : gen_markers[j] + len(pred)] = curr_var
                        gen_markers[j] += len(pred)

                    decoder_attention_mask = torch.ones_like(generated_seqs).float().to(generated_seqs.device)
                    decoder_attention_mask[generated_seqs == PAD_ID] = 0.0

                    decoder_output  = self.bert_decoder(input_ids=generated_seqs, attention_mask=decoder_attention_mask,
                                            encoder_hidden_states=encoder_output[i].unsqueeze(0), encoder_attention_mask=encoder_attention_mask[i].unsqueeze(0))[0]

                    probabilities   = F.log_softmax(self.fc_final(decoder_output), dim=-1)

                    # Add up the scores of the token at the __next__ time step

                    scores          = torch.zeros(generated_seqs.shape[0]).to(generated_seqs.device)
                    active          = torch.ones(generated_seqs.shape[0]).long().to(generated_seqs.device)
                    temp_markers    = orig_markers

                    while torch.sum(active) != 0:
                        position_probs  = torch.gather(probabilities, 1, (temp_markers - 1).reshape(-1, 1, 1).repeat(1, 1, probabilities.shape[2])).squeeze(1)
                        curr_tokens     = torch.gather(generated_seqs, 1, temp_markers.unsqueeze(1))
                        tok_probs       = torch.gather(position_probs, 1, curr_tokens).squeeze(1)

                        tok_probs       *= active
                        scores          += tok_probs

                        active          *= (temp_markers != (gen_markers-1)).long()
                        temp_markers    += active

                    # Update the prior probabilities
                    prior_probs         = prior_probs + scores

                else:
                    # You encounter a new variable which hasn't been seen before
                    # Generate <beam_width> possibilities for its name
                    generated_seqs, gen_markers, gen_var_ids, prior_probs, candidate_maps   = self.beam_search(generated_seqs, gen_markers, gen_var_ids, prior_probs, candidate_maps, curr_var,
                                                                                                   encoder_output[i].unsqueeze(0), encoder_attention_mask[i].unsqueeze(0), beam_width=5, top_k=self.top_k)

                source_marker += 1

            if top5 is True:
                final_inds  = torch.argsort(prior_probs, descending=True)[:5]
                confs       = torch.sort(prior_probs, descending=True)[0][:5].cpu().tolist()
                pred_maps   = [candidate_maps[f.item()] for f in final_inds]

                if len(pred_maps) == 1:
                    pred_maps = pred_maps * 5
                    confs = confs * 5

                batch_preds.append(pred_maps)
                batch_confs.append(confs)

            else:

                final_ind   = torch.argmax(prior_probs)
                conf        = torch.max(prior_probs).item()
                pred_map    = candidate_maps[final_ind]

                batch_preds.append(pred_map)
                batch_confs.append(conf)

        return batch_preds, batch_confs

    def beam_search(self, generated_seqs, gen_markers, gen_var_ids, prior_probs, candidate_maps, curr_var, encoder_output, encoder_attention_mask, beam_width=5, top_k=10):

        if generated_seqs.shape[0] * beam_width < top_k:
            beam_width = top_k

        active      = torch.ones_like(gen_markers).to(gen_markers.device)
        beam_alpha  = 0.7
        end_token   = self.vocab.all_subtokens.word2id['</s>']

        candidate_maps = candidate_maps
        orig_markers    = gen_markers.clone()

        for _ in range(10): # Predict at most 10 subtokens
            decoder_attention_mask = torch.ones_like(generated_seqs).float().to(generated_seqs.device)
            decoder_attention_mask[generated_seqs == PAD_ID] = 0.0

            decoder_output  = self.bert_decoder(input_ids=generated_seqs, attention_mask=decoder_attention_mask,
                                                encoder_hidden_states=encoder_output, encoder_attention_mask=encoder_attention_mask)[0]
            probabilities   = F.log_softmax(self.fc_final(decoder_output), dim=-1)
            # Gather the predictions at the current markers
            # (gen_marker - 1) because prediction happens one step ahead
            probabilities   = torch.gather(probabilities, 1, (gen_markers - 1).reshape(-1, 1, 1).repeat(1, 1, probabilities.shape[2])).squeeze(1)

            probs, preds    = probabilities.sort(dim=-1, descending=True)

            probs               *= active.unsqueeze(1)  # Set log prob of non-active ones to 0
            preds[active == 0]  = end_token  # Set preds of non-active ones to the end token (ie, remain unchanged)

            # Repeat active ones only once. Repeat the rest beam_width no. of times.
            filter_mask = torch.ones((preds.shape[0], beam_width)).long().to(preds.device)
            filter_mask *= active.unsqueeze(1)
            filter_mask[:, 0][active == 0] = 1
            filter_mask = filter_mask.reshape(-1)

            preds           = preds[:, :beam_width].reshape(-1)[filter_mask == 1]
            probs           = probs[:, :beam_width].reshape(-1)[filter_mask == 1]

            generated_seqs  = torch.repeat_interleave(generated_seqs,   beam_width, dim=0)[filter_mask == 1]
            gen_var_ids     = torch.repeat_interleave(gen_var_ids,      beam_width, dim=0)[filter_mask == 1]
            orig_markers    = torch.repeat_interleave(orig_markers,     beam_width, dim=0)[filter_mask == 1]
            gen_markers     = torch.repeat_interleave(gen_markers,      beam_width, dim=0)[filter_mask == 1]
            active          = torch.repeat_interleave(active,           beam_width, dim=0)[filter_mask == 1]
            prior_probs     = torch.repeat_interleave(prior_probs,      beam_width, dim=0)[filter_mask == 1]

            candidate_maps = [item.copy() for item in candidate_maps for _ in range(beam_width)]
            candidate_maps = [candidate_maps[i] for i in range(len(candidate_maps)) if filter_mask[i] == 1]

            generated_seqs.scatter_(1, gen_markers.unsqueeze(1), preds.unsqueeze(1))
            gen_var_ids.scatter_(1, gen_markers.unsqueeze(1), curr_var * torch.ones(preds.shape[0], 1).long().to(gen_var_ids.device))

            # lengths       = (gen_markers - gen_marker + 1).float()
            # penalties     = torch.pow(5 + lengths, beam_alpha) / math.pow(6, beam_alpha)
            penalties       = torch.ones_like(probs).to(probs.device)

            updated_probs   = probs + prior_probs

            sort_inds       = (updated_probs / penalties).argsort(descending=True)
            updated_probs   = updated_probs[sort_inds]

            prior_probs     = updated_probs[:top_k]

            new_preds       = preds         [sort_inds[:top_k]]
            generated_seqs  = generated_seqs[sort_inds[:top_k]]
            gen_var_ids     = gen_var_ids   [sort_inds[:top_k]]
            gen_markers     = gen_markers   [sort_inds[:top_k]]
            active          = active        [sort_inds[:top_k]]
            orig_markers    = orig_markers  [sort_inds[:top_k]]

            candidate_maps = [candidate_maps[ind.item()] for ind in sort_inds[:top_k]]

            active          = active * (new_preds != end_token).long()
            gen_markers     += active

            if torch.sum(active) == 0: break

        # gen_markers are pointing at the end_token. Move them one ahead
        gen_markers += 1

        assert generated_seqs.shape[0] == top_k

        for i in range(top_k):
            candidate_maps[i][curr_var] = generated_seqs[i][orig_markers[i] : gen_markers[i]].cpu().tolist()

        return generated_seqs, gen_markers, gen_var_ids, prior_probs, candidate_maps


class RenamingModelHybrid(nn.Module):

    def __init__(self, vocab, top_k, config, device):
        super(RenamingModelHybrid, self).__init__()

        self.vocab = vocab
        self.top_k = top_k
        self.source_vocab_size = len(self.vocab.source_tokens) + 1

        self.graph_encoder  = GraphASTEncoder.build(config['encoder']['graph_encoder'])
        self.graph_emb_size = config['encoder']['graph_encoder']['gnn']['hidden_size']
        self.emb_size       = 256

        state_dict = torch.load('saved_checkpoints/bert_2604/bert_pretrained_epoch_23_batch_140000.pth', map_location=device)

        keys_to_delete = ["cls.predictions.bias", "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight",
                        "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight", "cls.predictions.decoder.bias",
                        "cls.seq_relationship.weight", "cls.seq_relationship.bias"]

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            if k in keys_to_delete: continue
            name = k[5:] # remove `bert.`
            new_state_dict[name] = v

        bert_config = BertConfig(vocab_size=self.source_vocab_size, max_position_embeddings=512, num_hidden_layers=6, hidden_size=self.emb_size, num_attention_heads=4)
        self.bert_encoder = BertModel(bert_config)
        self.bert_encoder.load_state_dict(new_state_dict)

        self.target_vocab_size = len(self.vocab.all_subtokens) + 1

        bert_config = BertConfig(vocab_size=self.target_vocab_size, max_position_embeddings=1000, num_hidden_layers=6, hidden_size=self.emb_size, num_attention_heads=4, is_decoder=True)
        self.bert_decoder = BertModel(bert_config)

        state_dict = torch.load('saved_checkpoints/bert_0905/bert_decoder_epoch_19_batch_220000.pth', map_location=device)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            if k in keys_to_delete: continue
            if 'crossattention' in k: continue
            name = k[5:] # remove `bert.`
            new_state_dict[name] = v

        for key in new_state_dict:
            self.bert_decoder.state_dict()[key].copy_(new_state_dict[key])

        self.enc_graph_map  = nn.Linear(self.emb_size + self.graph_emb_size, self.emb_size)
        self.fc_final       = nn.Linear(self.emb_size, self.target_vocab_size)

        self.fc_final.weight.data = state_dict['model']['cls.predictions.decoder.weight']

    def forward(self, src_tokens, src_mask, variable_ids, target_tokens, graph_input):
        encoder_attention_mask = torch.ones_like(src_tokens).float().to(src_tokens.device)
        encoder_attention_mask[src_tokens == PAD_ID] = 0.0

        assert torch.max(src_tokens) < self.source_vocab_size
        assert torch.min(src_tokens) >= 0
        assert torch.max(target_tokens) < self.target_vocab_size
        assert torch.min(target_tokens) >= 0

        encoder_output  = self.bert_encoder(input_ids=src_tokens, attention_mask=encoder_attention_mask)[0]

        graph_output    = self.graph_encoder(graph_input)
        variable_emb    = graph_output['variable_encoding']

        graph_embedding = torch.gather(variable_emb, 1, variable_ids.unsqueeze(2).repeat(1, 1, variable_emb.shape[2])) * src_mask.unsqueeze(2)

        full_enc_output = self.enc_graph_map(torch.cat((encoder_output, graph_embedding), dim=2))

        decoder_attention_mask = torch.ones_like(target_tokens).float().to(target_tokens.device)
        decoder_attention_mask[target_tokens == PAD_ID] = 0.0

        decoder_output = self.bert_decoder(input_ids=target_tokens, attention_mask=decoder_attention_mask,
                                           encoder_hidden_states=full_enc_output, encoder_attention_mask=encoder_attention_mask)[0]

        predictions    = self.fc_final(decoder_output)

        return predictions

    def predict(self, src_tokens, src_mask, variable_ids, graph_input, approx=False):
        end_token   = self.vocab.all_subtokens.word2id['</s>']
        start_token = self.vocab.all_subtokens.word2id['<s>']
        batch_size  = src_tokens.shape[0]

        encoder_attention_mask = torch.ones_like(src_tokens).float().to(src_tokens.device)
        encoder_attention_mask[src_tokens == PAD_ID] = 0.0

        assert torch.max(src_tokens) < self.source_vocab_size
        assert torch.min(src_tokens) >= 0

        encoder_output          = self.bert_encoder(input_ids=src_tokens, attention_mask=encoder_attention_mask)[0]

        graph_output    = self.graph_encoder(graph_input)
        variable_emb    = graph_output['variable_encoding']

        graph_embedding = torch.gather(variable_emb, 1, variable_ids.unsqueeze(2).repeat(1, 1, variable_emb.shape[2])) * src_mask.unsqueeze(2)

        full_enc_output = self.enc_graph_map(torch.cat((encoder_output, graph_embedding), dim=2))

        source_vocab_to_target  = {self.vocab.source_tokens.word2id[t] : self.vocab.all_subtokens.word2id[t] for t in self.vocab.source_tokens.word2id.keys()}
        src_target_maps         = []
        confidences             = []

        for i in range(batch_size):

            if src_tokens[i][0] != start_token:
                input_sequence      = torch.zeros(src_tokens.shape[1] + 1).to(src_tokens.device)
                input_mask          = torch.zeros(src_mask.shape[1] + 1).to(src_mask.device)
                input_sequence[1:]  = src_tokens[i]
                input_mask[1:]      = src_mask[i]
            else:
                input_sequence      = src_tokens[i]
                input_mask          = src_mask[i]

            num_vars        = int(input_mask.sum())
            seq_len         = torch.sum((input_sequence != PAD_ID).long())
            generated_seqs  = torch.zeros(1, min(seq_len + 10 * num_vars, 1000)).long().to(src_tokens.device)

            source_marker   = 0
            gen_markers     = torch.LongTensor ([0]).to(generated_seqs.device)
            prior_probs     = torch.FloatTensor([0]).to(generated_seqs.device)

            candidate_maps = [{}]

            for _ in range(num_vars):
                # Filling up the known (non-identifier) tokens
                while source_marker < seq_len and input_mask[source_marker] != 1:
                    token           = input_sequence[source_marker]
                    values          = source_vocab_to_target[token.item()] * torch.ones_like(gen_markers).to(generated_seqs.device)

                    generated_seqs  = torch.scatter(generated_seqs, 1, gen_markers.unsqueeze(1), values.unsqueeze(1))

                    source_marker   += 1
                    gen_markers     += 1

                if source_marker >= seq_len: break

                curr_var = input_sequence[source_marker].item()

                if curr_var in candidate_maps[0]:
                    if approx is True:
                        source_marker += 1
                        continue
                    # If we've seen this variable before, just use the previous predictions and update the scores
                    # Note - it's enough to check candidate_maps[0] because if it is in the first map, it is in all of them

                    orig_markers = gen_markers.clone()

                    for j in range(len(candidate_maps)):
                        pred = candidate_maps[j][curr_var]
                        generated_seqs[j][gen_markers[j] : gen_markers[j] + len(pred)] = torch.LongTensor(pred).to(generated_seqs.device)
                        gen_markers[j] += len(pred)

                    decoder_attention_mask = torch.ones_like(generated_seqs).float().to(generated_seqs.device)
                    decoder_attention_mask[generated_seqs == PAD_ID] = 0.0

                    decoder_output  = self.bert_decoder(input_ids=generated_seqs, attention_mask=decoder_attention_mask,
                                            encoder_hidden_states=full_enc_output[i].unsqueeze(0), encoder_attention_mask=encoder_attention_mask[i].unsqueeze(0))[0]

                    probabilities   = F.log_softmax(self.fc_final(decoder_output), dim=-1)

                    # Add up the scores of the token at the __next__ time step

                    scores          = torch.zeros(generated_seqs.shape[0]).to(generated_seqs.device)
                    active          = torch.ones(generated_seqs.shape[0]).long().to(generated_seqs.device)
                    temp_markers    = orig_markers

                    while torch.sum(active) != 0:
                        position_probs  = torch.gather(probabilities, 1, (temp_markers - 1).reshape(-1, 1, 1).repeat(1, 1, probabilities.shape[2])).squeeze(1)
                        curr_tokens     = torch.gather(generated_seqs, 1, temp_markers.unsqueeze(1))
                        tok_probs       = torch.gather(position_probs, 1, curr_tokens).squeeze(1)

                        tok_probs       *= active
                        scores          += tok_probs

                        active          *= (temp_markers != (gen_markers-1)).long()
                        temp_markers    += active

                    # Update the prior probabilities
                    prior_probs         = prior_probs + scores

                else:
                    # You encounter a new variable which hasn't been seen before
                    # Generate <beam_width> possibilities for its name
                    generated_seqs, gen_markers, prior_probs, candidate_maps    = self.beam_search(generated_seqs, gen_markers, prior_probs, candidate_maps, curr_var,
                                                                                                   full_enc_output[i].unsqueeze(0), encoder_attention_mask[i].unsqueeze(0), beam_width=5, top_k=self.top_k)

                source_marker += 1

            final_ind       = torch.argmax(prior_probs)
            confidence      = torch.max(prior_probs).item()
            src_target_map  = candidate_maps[final_ind]

            src_target_maps.append(src_target_map)
            confidences.append(confidence)

        return src_target_maps, confidences

    def beam_search(self, generated_seqs, gen_markers, prior_probs, candidate_maps, curr_var, full_enc_output, encoder_attention_mask, beam_width=5, top_k=10):

        if generated_seqs.shape[0] * beam_width < top_k:
            beam_width = top_k

        active      = torch.ones_like(gen_markers).to(gen_markers.device)
        beam_alpha  = 0.7
        end_token   = self.vocab.all_subtokens.word2id['</s>']

        candidate_maps = candidate_maps
        orig_markers    = gen_markers.clone()

        for _ in range(10): # Predict at most 10 subtokens
            decoder_attention_mask = torch.ones_like(generated_seqs).float().to(generated_seqs.device)
            decoder_attention_mask[generated_seqs == PAD_ID] = 0.0

            decoder_output  = self.bert_decoder(input_ids=generated_seqs, attention_mask=decoder_attention_mask,
                                                encoder_hidden_states=full_enc_output, encoder_attention_mask=encoder_attention_mask)[0]
            probabilities   = F.log_softmax(self.fc_final(decoder_output), dim=-1)
            # Gather the predictions at the current markers
            # (gen_marker - 1) because prediction happens one step ahead
            probabilities   = torch.gather(probabilities, 1, (gen_markers - 1).reshape(-1, 1, 1).repeat(1, 1, probabilities.shape[2])).squeeze(1)

            probs, preds    = probabilities.sort(dim=-1, descending=True)

            probs               *= active.unsqueeze(1)  # Set log prob of non-active ones to 0
            preds[active == 0]  = end_token  # Set preds of non-active ones to the end token (ie, remain unchanged)

            # Repeat active ones only once. Repeat the rest beam_width no. of times.
            filter_mask = torch.ones((preds.shape[0], beam_width)).long().to(preds.device)
            filter_mask *= active.unsqueeze(1)
            filter_mask[:, 0][active == 0] = 1
            filter_mask = filter_mask.reshape(-1)

            preds           = preds[:, :beam_width].reshape(-1)[filter_mask == 1]
            probs           = probs[:, :beam_width].reshape(-1)[filter_mask == 1]

            generated_seqs  = torch.repeat_interleave(generated_seqs,   beam_width, dim=0)[filter_mask == 1]
            orig_markers    = torch.repeat_interleave(orig_markers,     beam_width, dim=0)[filter_mask == 1]
            gen_markers     = torch.repeat_interleave(gen_markers,      beam_width, dim=0)[filter_mask == 1]
            active          = torch.repeat_interleave(active,           beam_width, dim=0)[filter_mask == 1]
            prior_probs     = torch.repeat_interleave(prior_probs,      beam_width, dim=0)[filter_mask == 1]

            candidate_maps = [item.copy() for item in candidate_maps for _ in range(beam_width)]
            candidate_maps = [candidate_maps[i] for i in range(len(candidate_maps)) if filter_mask[i] == 1]

            generated_seqs.scatter_(1, gen_markers.unsqueeze(1), preds.unsqueeze(1))

            # lengths       = (gen_markers - gen_marker + 1).float()
            # penalties     = torch.pow(5 + lengths, beam_alpha) / math.pow(6, beam_alpha)
            penalties       = torch.ones_like(probs).to(probs.device)

            updated_probs   = probs + prior_probs

            sort_inds       = (updated_probs / penalties).argsort(descending=True)
            updated_probs   = updated_probs[sort_inds]

            prior_probs     = updated_probs[:top_k]

            new_preds       = preds         [sort_inds[:top_k]]
            generated_seqs  = generated_seqs[sort_inds[:top_k]]
            gen_markers     = gen_markers   [sort_inds[:top_k]]
            active          = active        [sort_inds[:top_k]]
            orig_markers    = orig_markers  [sort_inds[:top_k]]

            candidate_maps = [candidate_maps[ind.item()] for ind in sort_inds[:top_k]]

            active          = active * (new_preds != end_token).long()
            gen_markers     += active

            if torch.sum(active) == 0: break

        # gen_markers are pointing at the end_token. Move them one ahead
        gen_markers += 1

        assert generated_seqs.shape[0] == top_k

        for i in range(top_k):
            candidate_maps[i][curr_var] = generated_seqs[i][orig_markers[i] : gen_markers[i]].cpu().tolist()

        return generated_seqs, gen_markers, prior_probs, candidate_maps
