import json, csv
import random
import torch
import pandas as pd

from config import *

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def load_data(args, input_vocab, output_vocab):
    if args.data_dir.endswith('conll2003'):
        data = load_conll2003(args, input_vocab, output_vocab)
    elif args.data_dir.endswith('people_dairy'):
        data = load_people_dairy(args, input_vocab, output_vocab)
    elif args.data_dir.endswith('ACL'):
        data = load_ACL(args, input_vocab, output_vocab)
    elif args.data_dir.endswith('cmeee'):
        data = load_cmeee(args, input_vocab, output_vocab)
    elif args.data_dir.endswith('msra'):
        data = load_msra(args, input_vocab, output_vocab)
    return data

def load_people_dairy(args, input_vocab, output_vocab):
    with open(args.vocab_dir + '/chin_b_vocab.txt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            input_vocab.addWord(line)
    with open(args.data_dir + '/people_dairy_2014.json', encoding='utf-8') as f:
        input_seq, output_seq = [], []
        for line in f:
            sample = json.loads(line.strip())
            input_seq.append([ch for ch in sample['text']])
            output_seq.append(sample['labels'])
            for tag in sample['labels']:
                output_vocab.addWord(tag)


    data = list(zip(input_seq, output_seq))


    train_size = int(0.7 * len(data))
    val_size = int(0.2 * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    input_seq_train, output_seq_train = zip(*train_data)
    input_seq_val, output_seq_val = zip(*val_data)
    input_seq_test, output_seq_test = zip(*test_data)

    data = {
        'input_train': list(input_seq_train),
        'output_train': list(output_seq_train),
        'train_num': len(input_seq_train),
        'input_val': list(input_seq_val),
        'output_val': list(output_seq_val),
        'val_num': len(input_seq_val),
        'input_test': list(input_seq_test),
        'output_test': list(output_seq_test),
        'test_num': len(input_seq_test)
    }

    return data

def load_msra(args, input_vocab, output_vocab):
    with open(args.vocab_dir + '/chin_b_vocab.txt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            input_vocab.addWord(line)
    with open(args.data_dir + '/msra.json', encoding='utf-8') as f:
        input_seq, output_seq = [], []
        cnt = 0
        for line in f:
            sample = json.loads(line.strip())
            input_seq.append([ch for ch in sample['text']])
            output_seq.append(sample['labels'])
            for tag in sample['labels']:
                output_vocab.addWord(tag)

    data = list(zip(input_seq, output_seq))


    train_size = int(0.76 * len(data))
    val_size = int(0.16 * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    input_seq_train, output_seq_train = zip(*train_data)
    input_seq_val, output_seq_val = zip(*val_data)
    input_seq_test, output_seq_test = zip(*test_data)


    data = {
        'input_train': list(input_seq_train),
        'output_train': list(output_seq_train),
        'train_num': len(input_seq_train),
        'input_val': list(input_seq_val),
        'output_val': list(output_seq_val),
        'val_num': len(input_seq_val),
        'input_test': list(input_seq_test),
        'output_test': list(output_seq_test),
        'test_num': len(input_seq_test)
    }

    return data

def load_cmeee(args, input_vocab, output_vocab):
    with open(args.vocab_dir + '/chin_b_vocab.txt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            input_vocab.addWord(line)
    with open(args.data_dir + '/cmeee.json', encoding='utf-8') as f:
        input_seq, output_seq = [], []
        for line in f:
            sample = json.loads(line.strip())
            input_seq.append([ch for ch in sample['text']])
            output_seq.append(sample['labels'])
            for tag in sample['labels']:
                output_vocab.addWord(tag)

    data = list(zip(input_seq, output_seq))


    train_size = int(0.66 * len(data))
    val_size = int(0.22 * len(data))

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    input_seq_train, output_seq_train = zip(*train_data)
    input_seq_val, output_seq_val = zip(*val_data)
    input_seq_test, output_seq_test = zip(*test_data)

    data = {
        'input_train': list(input_seq_train),
        'output_train': list(output_seq_train),
        'train_num': len(input_seq_train),
        'input_val': list(input_seq_val),
        'output_val': list(output_seq_val),
        'val_num': len(input_seq_val),
        'input_test': list(input_seq_test),
        'output_test': list(output_seq_test),
        'test_num': len(input_seq_test)
    }

    return data

def load_ACL(args, input_vocab, output_vocab):
    with open(args.vocab_dir + '/chin_b_vocab.txt', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            input_vocab.addWord(line)
    
    input_seq_train, output_seq_train = [], []
    with open(args.data_dir + '/train.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        data = [(eval(row[0]), eval(row[1])) for row in reader]
        for sen, label in data:
            input_seq_train.append(sen)
            output_seq_train.append(label)
            for tag in label:
                output_vocab.addWord(tag)

    input_seq_val, output_seq_val = [], []
    with open(args.data_dir + '/dev.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        data = [(eval(row[0]), eval(row[1])) for row in reader]
        for sen, label in data:
            input_seq_val.append(sen)
            output_seq_val.append(label)

    input_seq_test, output_seq_test = [], []
    with open(args.data_dir + '/test.csv', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        data = [(eval(row[0]), eval(row[1])) for row in reader]
        for sen, label in data:
            input_seq_test.append(sen)
            output_seq_test.append(label)


    data = {
        'input_train': input_seq_train,
        'output_train': output_seq_train,
        'train_num': len(input_seq_train),
        'input_val': input_seq_val,
        'output_val': output_seq_val,
        'val_num': len(input_seq_val),
        'input_test': input_seq_test,
        'output_test': output_seq_test,
        'test_num': len(input_seq_test)
    }
    return data

def load_conll2003(args, input_vocab, output_vocab):
    input_seq_train, output_seq_train = [], []
    _input_seq, _output_seq = [], []
    with open(args.data_dir + '/train.txt', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                input_seq_train.append(_input_seq)
                output_seq_train.append(_output_seq)
                _input_seq, _output_seq = [], []
                continue
            if not line.startswith('-DOCSTART-'):
                items = line.rstrip().split()
                char, tag = items[0], items[-1]
                input_vocab.addWord(char)
                output_vocab.addWord(tag)
                _input_seq.append(char)
                _output_seq.append(tag)

    input_seq_val, output_seq_val = [], []
    _input_seq, _output_seq = [], []
    with open(args.data_dir + '/dev.txt', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                input_seq_val.append(_input_seq)
                output_seq_val.append(_output_seq)
                _input_seq, _output_seq = [], []
                continue
            if not line.startswith('-DOCSTART-'):
                items = line.rstrip().split()
                char, tag = items[0], items[-1]
                input_vocab.addWord(char)
                output_vocab.addWord(tag)
                _input_seq.append(char)
                _output_seq.append(tag)
    

    input_seq_test, output_seq_test = [], []
    _input_seq, _output_seq = [], []
    with open(args.data_dir + '/test.txt', encoding='utf-8') as f:
        for line in f:
            if line == '\n':
                input_seq_test.append(_input_seq)
                output_seq_test.append(_output_seq)
                _input_seq, _output_seq = [], []
                continue
            if not line.startswith('-DOCSTART-'):
                items = line.rstrip().split()
                char, tag = items[0], items[-1]
                #             input_vocab.addWord(char)
                #             output_vocab.addWord(tag)
                _input_seq.append(char)
                _output_seq.append(tag)


    data = {
        'input_train': input_seq_train,
        'output_train': output_seq_train,
        'train_num': len(input_seq_train),
        'input_val': input_seq_val,
        'output_val': output_seq_val,
        'val_num': len(input_seq_val),
        'input_test': input_seq_test,
        'output_test': output_seq_test,
        'test_num': len(input_seq_test)
    }
    return data


def data_iter(args, data, input_vocab, output_vocab, data_type, max_len):
    '''
    生成器函数，每次产生一个batch的数据
    '''

    if data_type == 'train':
        tmp_input = data['input_train'][:]
        tmp_output = data['output_train'][:]
    elif data_type == 'val':
        tmp_input = data['input_val'][:]
        tmp_output = data['output_val'][:]
    elif data_type == 'test':
        tmp_input = data['input_test'][:]
        tmp_output = data['output_test'][:]

    tmp_lens = list(map(lambda x: min(max_len, len(x)), tmp_input))
    lens_series = pd.Series(tmp_lens)
    sorted_idx = lens_series.sort_values(ascending=False).index

    tmp_input = pd.Series(tmp_input)
    tmp_output = pd.Series(tmp_output)
    tmp_lens = pd.Series(tmp_lens)

    sorted_input = tmp_input[sorted_idx]
    sorted_output = tmp_output[sorted_idx]
    sorted_lens = tmp_lens[sorted_idx]

    inputs, outputs = [], []
    for _len, _input, _output in zip(sorted_lens, sorted_input, sorted_output):
        inputs.append(_input[:_len])
        outputs.append(_output[:_len])

    src = []
    src_len = []
    trg = []

    input_unk_token_id = input_vocab.word2index['UNK']
    if output_vocab:
        output_unk_token_id = output_vocab.word2index['UNK']

    srcs = []
    src_lens = []
    trgs = []
    for seq_len, eit, dot in zip(sorted_lens, inputs, outputs):

        eit = ['SOS'] + eit
        dot = ['SOS'] + dot

        input_ids = torch.tensor(
            [input_vocab.word2index.get(i, input_unk_token_id) for i in eit])
        output_ids = torch.tensor(
            [output_vocab.word2index.get(i, output_unk_token_id) for i in dot])

        src.append(input_ids)
        src_len.append(seq_len + 2)  # 加了SOS和EOS
        trg.append(output_ids)

        if len(src) == args.batch_size:
            src = torch.nn.utils.rnn.pad_sequence(
                src, padding_value=input_vocab.word2index['PAD'])
            trg = torch.nn.utils.rnn.pad_sequence(
                trg, padding_value=output_vocab.word2index['PAD'])
            eosIndex = output_vocab.word2index['EOS']
            src = torch.cat((src, torch.tensor(eosIndex).repeat(src.shape[1]).unsqueeze(0)), dim = 0)
            trg = torch.cat((trg, torch.tensor(eosIndex).repeat(trg.shape[1]).unsqueeze(0)), dim = 0)
            srcs.append(src)
            src_lens.append(torch.tensor(src_len))
            trgs.append(trg)
            src, src_len, trg = [], [], []

    if len(src) > 0:  # 最后一个batch
        src = torch.nn.utils.rnn.pad_sequence(
            src, padding_value=input_vocab.word2index['PAD'])
        trg = torch.nn.utils.rnn.pad_sequence(
            trg, padding_value=output_vocab.word2index['PAD'])
        eosIndex = output_vocab.word2index['EOS']
        src = torch.cat((src, torch.tensor(eosIndex).repeat(src.shape[1]).unsqueeze(0)), dim = 0)
        trg = torch.cat((trg, torch.tensor(eosIndex).repeat(trg.shape[1]).unsqueeze(0)), dim = 0)
        srcs.append(src)
        src_lens.append(torch.tensor(src_len))
        trgs.append(trg)
    
    
    iters = list(zip(srcs, src_lens, trgs))
    random.shuffle(iters)
    for src, src_len, trg in iters:
        yield src, src_len, trg