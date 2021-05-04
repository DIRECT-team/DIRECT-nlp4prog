from helper import *

from utils.vocab import PAD_ID, Vocab

vocab = Vocab.load('data/vocab.bpe10000/vocab')

def tokens_to_word(inp_seq):
    output = ''
    for t in inp_seq:
        c = vocab.all_subtokens.id2word[t]
        if c == '<s>': c = ''
        if c == '</s>': c = ''
        if c == '<pad>': c = ''
        output += c
    return output

class HistogramBins(object):

    def __init__(self, thresholds, key_func):
        self.thresholds = np.array(thresholds)
        self.thresholds = np.concatenate((self.thresholds, [math.inf]))
        self.key_func   = key_func

        self.bins           = {thresh : [] for thresh in self.thresholds}

    def process(self, data):
        for x in data:
            key = self.key_func(x)

            bin_id = np.argmax((self.thresholds - key) > 0)
            self.bins[self.thresholds[bin_id]].append(x)

parser = argparse.ArgumentParser(description='Decompiled Identifier renaming')

parser.add_argument('-fname', dest="fname", required=True, help='Path to the file where the prediction results are dumped')

args = parser.parse_args()

with open('predictions/{}'.format(args.fname), 'rb') as f:
    all_preds = pickle.load(f)

examples = []

correct_body_not_in_train   = 0
total_body_not_in_train     = 0
correct_top5                = 0

total_edist     = 0
total_length    = 0
total_jaccard   = 0

for pred in all_preds:
    src_tokens, src_mask, pred_maps, target_map, result_dict, confidence, body_in_train = pred

    if isinstance(pred_maps, dict): pred_maps = [pred_maps]

    if body_in_train == 1:
        continue
    else:
        length  = np.sum(np.array(src_tokens) != 0)

        n_before = correct_body_not_in_train
        d_before = total_body_not_in_train

        for var in target_map:
            old_name = vocab.source_tokens.id2word[var]

            pred_names = []

            for p in pred_maps:
                if var not in p:
                    if var == vocab.source_tokens.word2id['<unk>']:
                        p[var] = []
                    else:
                        raise Exception("{} not in predicted variables".format(var))

                pred_names.append(tokens_to_word(p[var]))

            actual_name = tokens_to_word(target_map[var])

            if old_name == actual_name: continue

            if pred_names[0] == actual_name:
                correct_body_not_in_train += 1

            if actual_name in pred_names:
                correct_top5 += 1

            total_body_not_in_train += 1

            edist = editdistance.eval(pred_names[0], actual_name)
            assert edist == nltk.edit_distance(pred_names[0], actual_name)

            total_edist += edist
            total_length += len(actual_name)

            jaccard = nltk.jaccard_distance(set(pred_names[0]), set(actual_name))
            total_jaccard += jaccard

        example = (length, correct_body_not_in_train - n_before, total_body_not_in_train - d_before)
        examples.append(example)

        try:
            assert example[1] == result_dict['correct_body_not_in_train']
            assert example[2] == result_dict['total_body_not_in_train']
        except:
            import pdb; pdb.set_trace()

acc     = float(correct_body_not_in_train)/(total_body_not_in_train + 1e-7)
top5    = float(correct_top5)/(total_body_not_in_train + 1e-7)
cer     = float(total_edist)/(total_length + 1e-7)
jaccard = float(total_jaccard/(total_body_not_in_train + 1e-7))

print("Acc : {:.4}, Top-5 : {:.4}, CER : {:.4}, Jaccard Dist : {:.4}".format(acc, top5, cer, jaccard))

Bins = HistogramBins([50, 100, 150, 200, 300, 400, 500, 750, 1000], lambda x : x[0])
Bins.process(examples)

accuracies  = {}
lengths     = {}

for thresh in Bins.bins.keys():
    accuracies[thresh]  = np.sum([x[1] for x in Bins.bins[thresh]]) / float(np.sum([x[2] for x in Bins.bins[thresh]]))
    lengths[thresh]     = len(Bins.bins[thresh])

import pdb; pdb.set_trace()
