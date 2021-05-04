from helper import *
from utils.vocab import PAD_ID, Vocab
from utils.dataset import Dataset
from utils import nn_util
from models import RenamingModel, RenamingModelHybrid, RenamingModelDecoderOnly

class Main(object):

    def get_batches(self, data, batch_size, split='train', shuffle=True):

        loader = data[split].batch_iterator(batch_size=batch_size,
                                  return_examples=False,
                                  return_prediction_target=True,
                                  config=self.config, progress=True, train=(split == 'train'), max_seq_len=(512 if (self.p.train or self.p.short_only) else -1),
                                  num_readers=self.p.num_readers, num_batchers=self.p.num_batchers,
                                  shuffle=shuffle, seed=self.p.seed, test_only=(split == 'test' or split == 'dev'))

        for batch in loader:
            if split == 'train':
                body_in_train = None
            else:
                body_in_train = torch.LongTensor([example['function_body_in_train'] for example in batch.tensor_dict['test_meta']])

            if self.p.hybrid:
                seq_input           = batch.tensor_dict['seq_encoder_input']
                graph_input         = batch.tensor_dict['graph_encoder_input']
                prediction_target   = batch.tensor_dict['prediction_target']

                yield (seq_input['src_code_tokens'], prediction_target['src_with_true_var_names'],
                        seq_input['variable_mention_mask'], seq_input['variable_mention_to_variable_id'],
                        prediction_target['src_with_true_var_mask'],
                        prediction_target['source_to_target_maps'], body_in_train, graph_input)

            else:
                yield (batch.tensor_dict['src_code_tokens'], batch.tensor_dict['prediction_target']['src_with_true_var_names'],
                        batch.tensor_dict['variable_mention_mask'], batch.tensor_dict['prediction_target']['src_with_true_var_mask'],
                        batch.tensor_dict['prediction_target']['source_to_target_maps'], body_in_train)

    def process_batch(self, batch):
        if self.p.hybrid:
            X, Y, src_mask, variable_ids, target_mask, src_target_maps, body_in_train, graph_input = batch
        else:
            X, Y, src_mask, target_mask, src_target_maps, body_in_train = batch

        X                   = X.long().to(self.device)
        Y                   = Y.long().to(self.device)
        src_mask            = src_mask.float().to(self.device)
        target_mask         = target_mask.float().to(self.device)

        if self.p.hybrid:
            variable_ids    = variable_ids.long().to(self.device)
            graph_input     = nn_util.to(graph_input, self.device)

            return (X, Y, src_mask, variable_ids, target_mask, src_target_maps, body_in_train, graph_input)
        else:
            return (X, Y, src_mask, target_mask, src_target_maps, body_in_train)

    def load_data(self):
        data = {}
        data['train']   = Dataset('data/preprocessed_data/train-shard-*.tar')
        data['dev']     = Dataset('data/preprocessed_data/dev.tar')
        data['test']    = Dataset('data/preprocessed_data/test.tar')

        return data

    def add_model(self):

        if self.p.hybrid:
            self.config = {
                    'train': {'unchanged_variable_weight': 0.1, 'buffer_size': 5000},
                    'encoder': {  'graph_encoder': { 'bpe_model_path': None,
                                  'connections': [ 'top_down',
                                                   'bottom_up',
                                                   'terminals',
                                                   'variable_master_nodes',
                                                   'func_root_to_arg'],
                                  'decoder_hidden_size': 256,
                                  'dropout': 0.2,
                                  'gnn': { 'hidden_size': 128,
                                           'layer_timesteps': [8],
                                           'residual_connections': {'0': [0]}},
                                  'init_with_seq_encoding': False,
                                  'node_content_embedding_size': 128,
                                  'node_syntax_type_embedding_size': 64,
                                  'node_type_embedding_size': 64,
                                  'vocab_file': 'data/vocab.bpe10000/vocab'},
                                'type': 'HybridEncoder'},
                    'data': {'vocab_file': 'data/vocab.bpe10000/vocab'}
                 }
            model = RenamingModelHybrid(self.vocab, self.p.top_k, self.config, self.device)

        else:

            self.config = {'train': {'unchanged_variable_weight': 0.1, 'buffer_size': 5000},
                      'encoder': {'type': 'SequentialEncoder'},
                      'data': {'vocab_file': 'data/vocab.bpe10000/vocab'}
                     }

            if self.p.enc:
                model = RenamingModel(self.vocab, self.p.top_k, self.config, self.device)
            else:
                model = RenamingModelDecoderOnly(self.vocab, self.p.top_k, self.config, self.device)

        if len(self.p.gpu) > 1 and self.device == torch.device('cuda'):
            model = nn.DataParallel(model)

        model = model.to(self.device)

        return model

    def __init__(self, args):
        self.p = args

        if not os.path.isdir(self.p.log_dir): os.mkdir(self.p.log_dir)
        if not os.path.isdir(self.p.save_dir): os.mkdir(self.p.save_dir)

        pprint(vars(self.p))
        self.logger = get_logger(self.p.name, self.p.log_dir)
        self.logger.info(vars(self.p))

        self.save_path = os.path.join(self.p.save_dir, self.p.name) + '.pth'

        if self.p.gpu != '-1':
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        def lr_func(epoch):
            if epoch < 10:
                return 1.0
            elif 10 <= epoch and epoch < 25:
                return 0.3
            else:
                return 0.1

        self.data       = self.load_data()
        self.vocab      = Vocab.load('data/vocab.bpe10000/vocab')
        self.model      = self.add_model()
        self.loss_fn    = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.optim      = torch.optim.Adam(self.model.parameters(), lr=self.p.lr)
        self.scheduler  = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_func, last_epoch=-1)

        self.curr_epoch = 0

        if self.p.restore: self.load_model(self.save_path)

    def save_model(self, path):
        self.logger.info('Saving model to path : {}'.format(path))
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save({'model': model_to_save.state_dict(), 'optim': self.optim.state_dict(), 'scheduler': self.scheduler.state_dict(), 'epoch': self.curr_epoch}, path)

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optim.load_state_dict(state_dict['optim'])
        if 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])
        else:
            print("WARNING - could not load scheduler state dict")
        self.curr_epoch = state_dict['epoch']

    def get_acc(self, Y_pred, Y):
        preds   = torch.argmax(Y_pred, dim=1)
        acc     = torch.mean((preds == Y).float())
        return acc

    def tokens_to_word(self, inp_seq, id2word):
        output = ''
        for t in inp_seq:
            c = id2word[t]
            if c == '<s>': c = ''
            if c == '</s>': c = ''
            if c == '<pad>': c = ''
            output += c
        return output

    def get_metric(self, pred_maps, src_target_maps, body_in_train, X=None, src_mask=None):
        correct_body_in_train       = []
        total_body_in_train         = []
        correct_body_not_in_train   = []
        total_body_not_in_train     = []

        unk = self.vocab.source_tokens.word2id['<unk>']

        for i in range(len(src_target_maps)):
            cbit = tbit = cbnit = tbnit = 0.

            for var in src_target_maps[i]:
                if var not in pred_maps[i]:
                    # If variable name is unknown, then it may have got split up in tokenization (eg - "@@cc@@" -> "@@","cc","@@")
                    # So the variable mask will not have a "1" at the positions of "@@","cc","@@".
                    # So no prediction will be made for "@@cc@@". Count it as an error and move on.
                    if var == unk:  pred_maps[i][unk] = []
                    else:
                        raise Exception("{} not in predicted variables".format(var))

                old_name = self.vocab.source_tokens.id2word[var]
                pred_name = self.tokens_to_word(pred_maps[i][var], self.vocab.all_subtokens.id2word)
                actual_name = self.tokens_to_word(src_target_maps[i][var], self.vocab.all_subtokens.id2word)

                # If the variable doesn't need renaming
                if old_name == actual_name: continue

                if pred_name == actual_name:
                    if body_in_train[i] == 1:   cbit    += 1
                    else:                       cbnit   += 1

                if body_in_train[i] == 1:   tbit    += 1
                else:                       tbnit   += 1

            correct_body_in_train.append(cbit)
            total_body_in_train.append(tbit)
            correct_body_not_in_train.append(cbnit)
            total_body_not_in_train.append(tbnit)

        return correct_body_in_train, total_body_in_train, correct_body_not_in_train, total_body_not_in_train

    def combine_preds(self, piece_maps, piece_confs, top5=False):

        # piece_confs ~ [num_pieces, batch_size, 5]

        num_pieces, batch_size = len(piece_maps), len(piece_maps[0])
        piece_confs = np.array(piece_confs)

        def get_merged_map(piece_conf):
            merged_map = {}

            if self.p.first_piece:
                indices = range(num_pieces)
            else:
                indices = np.argsort(piece_conf)[::-1]

            for piece_ind in indices:

                new_map     = piece_maps[piece_ind][i][k]
                # Since this new map is lower in priority than all the previously considered ones,
                # we only incorporate new (hitherto unseen) variables into our merged map
                new_keys    = set(new_map.keys()).difference(set(merged_map.keys()))

                if len(new_keys) == 0: continue

                merged_map.update({key : new_map[key] for key in new_keys})

            return merged_map

        merged_maps = []

        for i in range(batch_size):

            if top5 is True:
                merged_batch_maps = []

                for k in range(5):

                    merged_map = get_merged_map(piece_confs[:, i, k])

                    merged_batch_maps.append(merged_map)

                merged_maps.append(merged_batch_maps)
        else:

            merged_map = get_merged_map(piece_confs[:, i])

        return merged_maps

    def predict(self, model, data, split, approx=False):
        model.eval()

        correct_body_in_train       = 0.
        total_body_in_train         = 0.
        correct_body_not_in_train   = 0.
        total_body_not_in_train     = 0.

        total_examples              = 0.

        all_data                    = []

        seq_lens        = []
        num_var         = []
        num_unique_var  = []

        agnostic_model = model.module if isinstance(model, nn.DataParallel) else model

        for step, batch in enumerate(self.get_batches(data, split=split, batch_size=self.p.batch_size, shuffle=True)):
            if self.p.hybrid:
                X, Y, src_mask, variable_ids, target_mask, src_target_maps, body_in_train, graph_input = self.process_batch(batch)
            else:
                X, Y, src_mask, target_mask, src_target_maps, body_in_train = self.process_batch(batch)

            seq_lens.extend([torch.sum(X[i] != 0).item() for i in range(X.shape[0])])
            num_var. extend([torch.sum(src_mask[i]).item() for i in range(src_mask.shape[0])])
            num_unique_var.extend([len(list(src_target_maps[i].keys())) for i in range(len(src_target_maps))])

            with torch.no_grad():
                if X.shape[1] > 512:
                    max_len     = X.shape[1]
                    piece_size  = 512
                    num_pieces  = int((max_len + piece_size - 1)/ piece_size)

                    piece_maps  = []
                    piece_confs = []


                    for p in range(num_pieces):
                        piece           = X             [:, p * piece_size : (p + 1) * piece_size]
                        mask_piece      = src_mask      [:, p * piece_size : (p + 1) * piece_size]

                        if self.p.hybrid:
                            var_id_piece = variable_ids [:, p * piece_size : (p + 1) * piece_size]
                            pred_maps, confidences  = agnostic_model.predict(piece, mask_piece, var_id_piece, graph_input, approx)
                        else:
                            pred_maps, confidences  = agnostic_model.predict(piece, mask_piece, approx, self.p.top5)

                        piece_maps. append(pred_maps)
                        piece_confs.append(confidences)

                    pred_maps = self.combine_preds(piece_maps, piece_confs, self.p.top5)

                else:
                    if self.p.hybrid:
                        pred_maps, confidences  = agnostic_model.predict(X, src_mask, variable_ids, graph_input, approx)
                    else:
                        pred_maps, confidences  = agnostic_model.predict(X, src_mask, approx, self.p.top5)

            if self.p.top5:
                best_pred_maps  = [p[0] for p in pred_maps]
            else:
                best_pred_maps  = pred_maps

            result          = self.get_metric(best_pred_maps, src_target_maps, body_in_train, X, src_mask)
            result          = np.array(result)

            correct_body_in_train       += np.sum(result[0])
            total_body_in_train         += np.sum(result[1])
            correct_body_not_in_train   += np.sum(result[2])
            total_body_not_in_train     += np.sum(result[3])

            total_examples              += X.shape[0]

            X        = X.cpu().tolist()
            src_mask = src_mask.cpu().tolist()

            for i in range(len(X)):
                result_dict =  {'correct_body_in_train'         : result[0][i],
                                'total_body_in_train'           : result[1][i],
                                'correct_body_not_in_train'     : result[2][i],
                                'total_body_not_in_train'       : result[3][i]
                                }

                all_data.append([X[i], src_mask[i], pred_maps[i], src_target_maps[i], result_dict, confidences[i], body_in_train[i]])

            if (step + 1) % 10 == 0:
                train_acc   = correct_body_in_train     / float(total_body_in_train + 1e-7)
                test_acc    = correct_body_not_in_train / float(total_body_not_in_train + 1e-7)

                self.logger.info("Prediction batch : {}, Total examples : {}, Train Acc : {:.4}, Test Acc : {:.4}".format(step, int(total_examples), train_acc, test_acc))
                self.logger.info("Correct train : {}, Total train : {}, Correct test : {}, Total test : {}".format(correct_body_in_train, total_body_in_train, correct_body_not_in_train, total_body_not_in_train))

                approx_indicator    = '_approx' if approx else ''
                short_indicator     = '_short'  if self.p.short_only else ''
                first_piece_ind     = '_piece1' if self.p.first_piece else ''
                val_ind             = '_val'    if self.p.val else ''

                if not os.path.isdir('predictions'): os.mkdir('predictions')

                with open('predictions/{}_epoch_{}_topk_{}{}{}{}{}.pkl'.format(self.p.name, self.curr_epoch, self.p.top_k, approx_indicator, short_indicator, first_piece_ind, val_ind), 'wb') as f:
                    pickle.dump(all_data, f)

        train_acc   = correct_body_in_train     / float(total_body_in_train + 1e-7)
        test_acc    = correct_body_not_in_train / float(total_body_not_in_train + 1e-7)

        print("Train Acc : {:.4}, Test Acc : {:.4}".format(train_acc, test_acc))

        return train_acc, test_acc

    def run_epoch(self, model, loss_fn, epoch_num, data, split='train'):
        model.train()
        losses, accs    = [], []
        var_penalty     = 50.0
        cum_loss        = 0.0
        num_examples    = 0

        for step, batch in enumerate(self.get_batches(data, split=split, batch_size=self.p.batch_size)):
            self.optim.zero_grad()

            if self.p.hybrid:
                X, Y, src_mask, variable_ids, target_mask, _, _, graph_input = self.process_batch(batch)
                Y_pred          = model.forward(X, src_mask, variable_ids, Y, graph_input)
            else:
                X, Y, _, target_mask, _, _ = self.process_batch(batch)
                Y_pred          = model.forward(X, Y)

            Y_pred          = Y_pred[:, :-1, :] # Remove last irrelevant prediction
            Y               = Y[:, 1:]          # Remove "start of sequence" token from label
            loss_unreduced  = loss_fn(Y_pred.permute(0, 2, 1), Y)

            if self.p.weights:
                penalty     = target_mask[:, 1:] * (var_penalty - 1) + 1
            elif self.p.only_var:
                penalty     = target_mask[:, 1:]
            else:
                penalty     = torch.ones_like(target_mask[:, 1:]).to(target_mask.device)

            penalty         /= penalty.mean()
            loss            = (loss_unreduced * penalty).mean(dim=-1)
            mean_loss       = loss.mean()

            mean_loss.backward()
            self.optim.step()

            losses.append(mean_loss.item())
            cum_loss        += loss.sum().item()
            num_examples    += X.shape[0]

            if step % 100 == 0:
                try:
                    curr_lr = self.scheduler.get_last_lr()[0]
                except:
                    curr_lr = self.scheduler.get_lr()[0]

                self.logger.info('E: {} \t Step : {} \t Loss: {:.5} \t LR: {:.5}'.format(epoch_num, step, cum_loss/num_examples, curr_lr))
                cum_loss = num_examples = 0

        return np.mean(losses)

    def fit(self):

        kill_count = 0

        while self.curr_epoch < self.p.num_epoch:
            self.curr_epoch += 1

            train_loss          = self.run_epoch(self.model, self.loss_fn, self.curr_epoch, self.data, split='train')

            self.scheduler.step()

            if self.p.save:     self.save_model(self.save_path)
            if self.p.save_all: self.save_model(self.save_path[:-4] + '_epoch_{}.pth'.format(self.curr_epoch))

            self.logger.info('[Epoch {}]: \t Training Loss : {}'.format(self.curr_epoch, train_loss))


    def eval(self, split='test', approx=False):
        train_acc, test_acc = self.predict(self.model, self.data, split, approx=approx)

        self.logger.info('[Evaluation]: {} Accuracy: {:.4}'.format(split, test_acc))


if __name__== "__main__":

    name_hash = 'test_' + str(uuid.uuid4())[:8]

    parser = argparse.ArgumentParser(description='Decompiled Identifier renaming')

    parser.add_argument('-name',        dest="name",        default=name_hash,              help='name')
    parser.add_argument('-restore',     dest="restore",     action='store_true',            help='Restore saved model')
    parser.add_argument('-train',       dest="train",       action='store_true',            help='To train or not to train')
    parser.add_argument('-val',         dest="val",         action='store_true',            help='Evaluate on validation set')
    parser.add_argument('-top1',        dest="top5",        action='store_false',           help='Return only top prediction')
    parser.add_argument('-logdir',      dest="log_dir",     default='log',  type=str,       help='log_dir')
    parser.add_argument('-nosave',      dest="save",        action='store_false',           help='Whether to save model')
    parser.add_argument('-save_dir',    dest="save_dir",    default='saved_checkpoints',    type=str,   help='save_dir')
    parser.add_argument('-save_all',    dest="save_all",    action='store_true',            help='Save model at every epoch')

    parser.add_argument('-gpu',         dest="gpu",         default='0',    type=str,       help='gpu')
    parser.add_argument('-seed',        dest="seed",        default=42,     type=int,       help='seed')
    parser.add_argument('-num_readers', dest="num_readers", default=10,     type=int,       help='num_readers')
    parser.add_argument('-num_batchers',dest="num_batchers",default=10,     type=int,       help='num_batchers')

    # Training parameters
    parser.add_argument('-batch',       dest="batch_size",  default=4096,   type=int,       help='batch_size')
    parser.add_argument('-lr',          dest="lr",          default=1e-4,   type=float,     help='lr')
    parser.add_argument('-num_epoch',   dest="num_epoch",   default=85,     type=int,       help='num_epoch')

    # Model parameters
    parser.add_argument('-hybrid',      dest="hybrid",      action='store_true',            help='To use graph or not')
    parser.add_argument('-no_enc',      dest="enc",         action='store_false',           help='To use encoder or not')
    parser.add_argument('-no_weights',  dest="weights",     action='store_false',           help='To use weights on variables in loss or not')
    parser.add_argument('-only_var',    dest="only_var",    action='store_true',            help='Weights only on variables in loss')

    # Prediction parameters
    parser.add_argument('-top_k',       dest="top_k",       default=2,      type=int,       help='Top k for Advanced Prediction')
    parser.add_argument('-approx',      dest="approx",      action='store_true',            help='Use pred at first occurrence of variable in seq')
    parser.add_argument('-conf_piece',  dest="first_piece", action='store_false',           help='For >512, if multiple pieces have a pred for var, use highest confidence')
    parser.add_argument('-short_only',  dest="short_only",  action='store_true',            help='Predict on <=512 sequences only')

    args = parser.parse_args()

    if not args.restore:
        args.name = args.name + '_' + time.strftime("%d-%m-%Y") + '_' + time.strftime("%H:%M:%S")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available() and args.gpu != '-1':
        print('Warning - no GPU available. Ignoring gpu parameter')
        args.gpu = '-1'

    set_gpu(args.gpu)

    main = Main(args)

    if args.train:
        main.fit()
    else:
        split = 'dev' if args.val else 'test'
        main.eval(split=split, approx=args.approx)
