import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import *

def get_input_ids(max_len, tokens, input_vocab, output_vocab):
    '''
        tokenize a sentence `tokens` according to `input_vocab`, return its token and corresponding length
    '''
    tokens = [i for i in tokens]
    seq_len = min(max_len, len(tokens))
    input_unk_token_id = input_vocab.word2index['UNK']

    tokens = tokens[:seq_len]
    tokens = ['SOS'] + tokens
    input_ids = torch.tensor([input_vocab.word2index.get(
        i, input_unk_token_id) for i in tokens]).unsqueeze(1)
    eosIndex = output_vocab.word2index['EOS']
    input_ids = torch.cat((input_ids, torch.tensor([eosIndex]).repeat(input_ids.shape[1]).unsqueeze(0)), dim = 0)
    return input_ids, seq_len + 2  # 加2是因为有'SOS'和'EOS'

def parse(max_len, tokens, input_vocab, output_vocab, model):
    ''' 
        将一个句子`tokens`解析为BIO标注
        Output: 
            trg_tokens: BIO标注列表，like: ['B-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O']
    '''
    model.eval()


    src, src_len = get_input_ids(max_len, tokens, input_vocab, output_vocab)

    src = src.permute(1, 0).to(device)
    src_len = torch.LongTensor([src_len]).to(device)

    with torch.no_grad():
        outputs = model(src, None, teacher_forcing_ratio = 0)
        if model.bm:  # beam_search时outputs就是BIO标注序列
            trg_ids = outputs[:-1]
        else:  # 不用beam_search时outputs是模型的输出，包含了BIO标注的概率分布，要用argmax取出最大概率对应的BIO标注
            trg_ids = outputs[0][1:-1].argmax(1)

    # if random.random() > .6:
    #     trg_tokens = [output_vocab.index2word[i.item()] for i in trg_ids]
    # else:
    #     trg_tokens = list(output_test)
    trg_tokens = [output_vocab.index2word[i.item()] for i in trg_ids]
    
    return trg_tokens

def parse_entities(sens, preds):
    '''
        生成器函数，从多个句子`sens`和对应的BIO标注预测`preds`中解析出每个句子的实体，每次产生一个句子的实体解析结果
    '''
    
    for sen, pred in zip(sens, preds):
        entities = []
        tmp = ''
        label = ''
        start_idx = 0
        begin_flag = False

        for i, (char, tag) in enumerate(zip(sen, pred)):
            if tag[0] == 'B':
                if tmp:
                    entities.append({'start_idx': start_idx, 'end_idx': i - 1, 'entity': tmp, 'label': label})
                    tmp = ''
                    label = ''
                tmp = char
                label = tag[2:]  # label: B-ORG -> ORG
                start_idx = i
                begin_flag = True
            elif tag[0] == 'I' and begin_flag:
                tmp += ' ' + char
            elif tag[0] == 'O' and begin_flag:
                if tmp:
                    entities.append({'start_idx': start_idx, 'end_idx': i - 1, 'entity': tmp, 'label': label})
                tmp = ''
                label = ''
                begin_flag = False
        if tmp:
            entities.append({'start_idx': start_idx, 'end_idx': i, 'entity': tmp, 'label': label})
        yield entities

def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) == list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list

def depict(args, val_losses, train_losses, val_accs, train_accs):
    '''
    绘制训练过程中`Loss`和`Acc`的变化曲线
    '''
    fig1, ax1 = plt.subplots()
    plt.plot(val_losses, label='valid')
    plt.plot(train_losses, label = 'train')
    plt.xticks(np.linspace(0, args.epochs, args.epochs // 100 + 1, dtype = int))
    ax1.set_title('Loss vs Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    plt.legend()

    fig2, ax2 = plt.subplots()
    plt.plot(val_accs, label = 'valid_acc')
    plt.plot(train_accs, label = 'train_acc')
    plt.xticks(np.linspace(0, args.epochs, args.epochs // 100 + 1, dtype = int))
    ax2.set_title('Acc vs Epochs')
    ax2.set_ylabel('Token-wise Accuracy')
    ax2.set_xlabel('Epochs')
    plt.legend()
    plt.show()
        
    fig1.savefig(args.pic_dir + '/loss.png')
    fig2.savefig(args.pic_dir + '/acc.png')

def epoch_time(start_time, end_time):
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs

def init_weights(model):
    for _, param in model.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
