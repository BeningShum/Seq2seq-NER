import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import math

from config import *

from data_loader import Vocab, load_data, data_iter
from utils import *
from model import *
from metrics import Metrics



def train_step(args, model, data, input_vocab, output_vocab, optimizer, criterion):
    '''
    Train the model for one epoch.
    '''
    
    model.train()
    
    epoch_loss = 0
    total_steps = 0
    n_trg = 0
    correct = 0
    iterator = data_iter(args,
        data,
        input_vocab,
        output_vocab,
        data_type='train',
        max_len = MAX_LEN)
    
    for batch in tqdm(iterator, desc='Training', total=math.ceil(data['train_num'] / args.batch_size)):
        
        batch = tuple(t.to(device) for t in batch)
        src, _, trg = batch
        if src.size(0) == 2:  # 单句长度为0不计算Loss
            continue
        
        optimizer.zero_grad()
        src = src.permute(1, 0)  # src shape: [batch_size, src_len]
        trg = trg.permute(1, 0)  # trg shape: [batch_size, trg_len]
        output = model(src, trg)  # output shape: [batch_size, trg_len, output_dim]
        
        
        output_dim = output.shape[-1]
        
        output_1 = output[:, 1:-1].contiguous().view(-1, output_dim)
        trg_1 = trg[:, 1:-1].contiguous().view(-1)
        

        mask = torch.logical_not(torch.eq(trg.long(), torch.tensor(0)))
        mask = mask.type(torch.bool)  # shape: [batch, seq_Len], 用来标记pad的位置，不计入accuracy的计算
        

        pred = torch.argmax(output, 2)
        
        pred = torch.masked_select(pred, mask)  # 经过mask_select, 被拉长为一维张量
        true = torch.masked_select(trg.long(), mask)
        n_trg += true.shape[0]

        correct += torch.sum(true == pred)
        
        loss = criterion(output_1, trg_1.long())
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        total_steps += 1


    return epoch_loss / total_steps , correct.item() / n_trg

def evaluate_step(args, model, data, input_vocab, output_vocab, criterion):
    '''
    在验证集上评估模型，返回loss和accuracy
    '''
    model.eval()
    model.bm = 0
    
    
    epoch_loss = 0
    total_steps = 0
    correct = 0
    n_tags = 0
    with torch.no_grad():
        val_iter = data_iter(args,
            data,
            input_vocab,
            output_vocab,
            data_type='val',
            max_len = MAX_LEN)         
        for batch in tqdm(val_iter, desc='Evaluating', total=math.ceil(data['val_num']/args.batch_size)):  

            batch = tuple(t.to(device) for t in batch)
            src, _, trg = batch
            if src.size(0) == 2:  # 单句长度为0不计算Loss
                continue

            src = src.permute(1, 0)
            trg = trg.permute(1, 0)                  
            output = model(src, trg, teacher_forcing_ratio = 0) # 关闭teacher forcing进行evaluate


            output_dim = output.shape[-1]

            output_1 = output[:, 1:-1].contiguous().view(-1, output_dim)
            trg_1 = trg[:, 1:-1].contiguous().view(-1)



            loss = criterion(output_1, trg_1.long())

            epoch_loss += loss.item()
            total_steps += 1

            mask = torch.logical_not(torch.eq(trg.long(), torch.tensor(0)))  # =>[batch, seq_Len], each element is a bool value, and if there is a pad in the sequence, the corresponding position of the mask is False
            mask = mask.type(torch.bool)
            
            pred = torch.argmax(output,2)
            pred = torch.masked_select(pred,mask)
            true = torch.masked_select(trg.long(),mask)
            correct += torch.sum(true == pred)
            n_tags += true.shape[0]

    
    model.bm = args.bm
    
    return epoch_loss / total_steps, correct.item() / n_tags

def train(args, model, data, input_vocab, output_vocab):
    '''
    模型训练主函数
    '''
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters.')


    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_TOKEN)


    val_losses = []
    train_losses = []
    val_accs = []
    train_accs = []
    best_valid_acc = 0

    for epoch in range(args.epochs):
        
        start_time = time.time()
        train_loss, train_acc = train_step(args, model, data, input_vocab, output_vocab, optimizer, criterion)
        valid_loss, valid_acc = evaluate_step(args, model, data, input_vocab, output_vocab, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | \t Train Acc: {train_acc: 3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} | \t Valid Acc: {valid_acc: 3f}')

        if valid_acc > best_valid_acc:  # 验证集上取得了更好的效果就保存模型

            best_valid_acc = valid_acc
            model_filename = args.model_dir + '/' + args.data_dir.split('/')[-1] + '_model.pt'
            torch.save(model.state_dict(), model_filename)
            print('Best model saved!')


        val_losses.append(valid_loss)
        val_accs.append(valid_acc)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

    return val_losses, train_losses, val_accs, train_accs

def evaluate(args, model, input_test, output_test, input_vocab, output_vocab, max_len):
    '''
    在测试集上评估模型性能
    '''
    preds = []
    trues = []
    for i, sen in tqdm(enumerate(input_test), desc='Testing', total=len(input_test)):
        if len(sen) <= max_len:
            trues.append(output_test[i])  # output_test[i]是当前sen对应的真实BIO标签序列
            pred = parse(max_len, sen, input_vocab, output_vocab, model)  # pred是预测的BIO标签序列
            preds.append(pred)
    
    Metric = Metrics(trues, preds, remove_O = True)  # 创建一个评估对象
    Metric.report_scores()  # 在终端输出评估结果
    Metric.save_ner_results(input_test, args.data_dir)  # 保存NER识别结果



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = "./data/ACL")
    parser.add_argument('--vocab_dir', type = str, default = "./vocabs")
    parser.add_argument('--model_dir', type = str, default = "./trained_models")
    parser.add_argument('--pic_dir', type = str, default = "./pics")  # 保存图片的目录
    parser.add_argument('--bm', type = int, default = 2)  # beam_search的束大小, 2 by default
    parser.add_argument('--attention_type', type = str, default = "Vaswani")  # 注意力类型，加性或点积
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 3e-5)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--evaluate', action = 'store_true')  # False表示训练，True表示评估
    args = parser.parse_args()


    input_vocab = Vocab(name = 'input')
    output_vocab = Vocab(name = 'output')
    data = load_data(args, input_vocab, output_vocab)

    INPUT_DIM = input_vocab.n_words
    OUTPUT_DIM = output_vocab.n_words

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT).to(device)
    attention = Attention(args, HID_DIM, ATTN_DROPOUT).to(device)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT).to(device)
    model = Seq2Seq(args, enc, dec, attention, device).to(device)

    if not args.evaluate:
        print('Train model on dataset: ' + args.data_dir.split('/')[-1] + '.')
        val_losses, train_losses, val_accs, train_accs = train(args, model, data, input_vocab, output_vocab)
        depict(args, val_losses, train_losses, val_accs, train_accs)
        model_filename = args.model_dir +'/' + args.data_dir.split('/')[-1] + '_model.pt'
        print('Train finished. Best model saved to ' + model_filename + '.')
    else:  # 如果是评估模式，加载模型进行评估
        try:
            model_filename = args.model_dir + '/' + args.data_dir.split('/')[-1] + '_model.pt'
            model_params = torch.load(model_filename)
            model.load_state_dict(model_params)
        except FileNotFoundError:
            raise FileNotFoundError('No model found, please train first or check the directory path!')
        print('Testing model from ' + model_filename + '...')
        evaluate(args, model, data['input_test'], data['output_test'], input_vocab, output_vocab, MAX_LEN)
        


if __name__ == '__main__':
    main()