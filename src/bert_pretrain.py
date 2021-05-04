from transformers import BertForPreTraining, BertConfig

from helper import *

from utils.dataset import Dataset
from utils.vocab import Vocab

import os

def mask_tokens(inputs, mask_token_id, vocab_size, mlm_probability=0.15):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args):

    if not os.path.exists(args.save_dir): os.mkdir(args.save_dir)

    if args.gpu != '-1' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    config = {'train': {'unchanged_variable_weight': 0.1, 'buffer_size': 5000},
                      'encoder': {'type': 'SequentialEncoder'},
                      'data': {'vocab_file': 'data/vocab.bpe10000/vocab'}
                    }

    train_set = Dataset('data/preprocessed_data/train-shard-*.tar')
    dev_set = Dataset('data/preprocessed_data/dev.tar')

    vocab = Vocab.load('data/vocab.bpe10000/vocab')

    if args.decoder:
        vocab_size = len(vocab.all_subtokens) + 1
    else:
        vocab_size = len(vocab.source_tokens) + 1

    max_iters   = args.max_iters
    lr          = args.lr
    warm_up     = args.warm_up

    batch_size  = 4096
    effective_batch_size = args.batch_size

    max_embeds = 1000 if args.decoder else 512

    bert_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_embeds, num_hidden_layers=6, hidden_size=256, num_attention_heads=4)
    model       = BertForPreTraining(bert_config)

    if args.restore:
        state_dict  = torch.load(os.path.join(args.save_dir, args.res_name))
        model.load_state_dict(state_dict['model'])
        batch_count = state_dict['step']
        epoch       = state_dict['epoch']

    model.train()
    model.to(device)

    if len(args.gpu) > 1 and device == torch.device('cuda'):
        model = nn.DataParallel(model)

    def lr_func(step):
        if step > warm_up:
            return (max_iters - step)/(max_iters - warm_up)
        else:
            return (step/warm_up)

    optimizer   = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-6, weight_decay=0.01)
    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, last_epoch=-1)
    loss_fn     = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    if args.restore:
        optimizer.load_state_dict(state_dict['optim'])
        scheduler.load_state_dict(state_dict['scheduler'])

    batch_count = 0
    epoch       = 0
    cum_loss    = 0.0

    while True:
        # load training dataset, which is a collection of ASTs and maps of gold-standard renamings
        train_set_iter = train_set.batch_iterator(batch_size=batch_size,
                                                  return_examples=False,
                                                  config=config, progress=True, train=True,
                                                  max_seq_len=512,
                                                  num_readers=args.num_readers, num_batchers=args.num_batchers)
        epoch += 1
        print("Epoch {}".format(epoch))

        loss = 0
        num_seq = 0

        optimizer.zero_grad()

        for batch in train_set_iter:
            if args.decoder:
                input_ids = batch.tensor_dict['prediction_target']['src_with_true_var_names']
            else:
                input_ids = batch.tensor_dict['src_code_tokens']

            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids == 0] = 0.0

            assert torch.max(input_ids) < vocab_size
            assert torch.min(input_ids) >= 0

            if input_ids.shape[0] > max_embeds:
                print("Warning - length {} is greater than max length {}. Skipping.".format(input_ids.shape[0], max_embeds))
                continue

            input_ids, labels = mask_tokens(inputs=input_ids, mask_token_id=vocab_size-1, vocab_size=vocab_size, mlm_probability=0.15)

            input_ids[attention_mask == 0]  = 0
            labels[attention_mask == 0]     = -100

            if torch.cuda.is_available():
                input_ids       = input_ids.cuda()
                labels          = labels.cuda()
                attention_mask  = attention_mask.cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, masked_lm_labels=labels)

            unreduced_loss  =  loss_fn(outputs[0].view(-1, bert_config.vocab_size), labels.view(-1)).reshape(labels.shape)/(torch.sum(labels != -100, axis=1).unsqueeze(1) + 1e-7)
            loss            += unreduced_loss.sum()
            num_seq         += input_ids.shape[0]

            if num_seq > effective_batch_size:
                batch_count += 1
                loss /= num_seq
                cum_loss += loss.item()

                if batch_count % 20 == 0:
                    print("{} batches, Loss : {:.4}, LR : {:.6}".format(batch_count, cum_loss/20, scheduler.get_lr()[0]))
                    cum_loss = 0.0

                if batch_count % 10000 == 0:
                    fname1  = os.path.join(args.save_dir, 'bert_{}_step_{}.pth'.format(('decoder' if args.decoder else 'encoder'), batch_count))
                    fname2  = os.path.join(args.save_dir, 'bert_{}.pth'.format(('decoder' if args.decoder else 'encoder'), batch_count))

                    state   = {'epoch' : epoch, 'step' : batch_count, 'model' : model.module.state_dict(), 'optim' : optimizer.state_dict(), 'scheduler' : scheduler.state_dict()}

                    torch.save(state, fname1)
                    torch.save(state, fname2)

                    print("Saved file to path {}".format(fname1))
                    print("Saved file to path {}".format(fname2))

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss = 0
                num_seq = 0

            if batch_count == max_iters:
                print(f'[Learner] Reached max iters', file=sys.stderr)
                exit()

        print("Max_len = {}".format(max_len))
        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Decompiled Identifier renaming')

    parser.add_argument('-seed',        dest="seed",        default=42,     type=int,       help='seed')
    parser.add_argument('-gpu',         dest="gpu",         default='0',    type=str,       help='gpu')

    parser.add_argument('-lr',          dest="lr",          default=1e-4,   type=float,     help='lr')
    parser.add_argument('-max_iters',   dest="max_iters",   default=100000, type=int,       help='max_iters')
    parser.add_argument('-warm_up',     dest="warm_up",     default=40000,  type=int,       help='warm_up')
    parser.add_argument('-batch',       dest="batch_size",  default=128,    type=int,       help='batch_size')

    parser.add_argument('-num_readers', dest="num_readers", default=10,     type=int,       help='num_readers')
    parser.add_argument('-num_batchers',dest="num_batchers",default=10,     type=int,       help='num_batchers')

    parser.add_argument('-restore',      dest="restore",      action='store_true',            help='restore')
    parser.add_argument('-decoder',      dest="decoder",      action='store_true',            help='decoder')
    parser.add_argument('-save_dir',    dest="save_dir",    default='saved_checkpoints',    type=str,   help='save_dir')

    args = parser.parse_args()

    set_gpu(args.gpu)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)
