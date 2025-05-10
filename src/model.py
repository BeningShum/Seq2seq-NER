import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from config import *

class Attention(nn.Module):
    '''
    Use Bahdanau/Vaswani attention mechanism for Seq2Seq.
    '''
    def __init__(self, args, hid_dim, dropout, key_dim=None, query_dim=None):
        super().__init__()
        
        query_dim = hid_dim if query_dim is None else query_dim
        key_dim = 2 * hid_dim if key_dim is None else key_dim
        
        self.type = args.attention_type
        self.hid_dim = hid_dim
        self.key_layer = nn.Linear(key_dim, hid_dim, bias=False)
        self.query_layer = nn.Linear(query_dim, hid_dim, bias=False)
        self.energy_layer = nn.Linear(hid_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        
        self.alphas = None  # to store attention scores
        
    def forward(self, query, key, value, mask):
        '''
        Args:
            query: [batch_size, hid_dim], previous hidden state of the final layer of the decoder LSTM or a zeros tensor.
            key: [batch_size, src_len, hid_dim * 2], output of the encoder .
            value: [batch_size, src_len, hid_dim * 2], same as key.
            mask: [batch_size, 1, src_len], mask out padding positions.
        Output:
            context: [batch_size, 1, hid_dim * 2], context extracted from the encoder for the current decoding step.
            alphas: [batch_size, 1, src_len], attention scores, not used in the model.
        '''

        assert mask is not None, "mask is required"

        proj_query = self.query_layer(query)
        proj_key = self.key_layer(key)

        if self.type == 'Bahdanau':
            scores = self.energy_layer(torch.tanh(proj_query.unsqueeze(1) + proj_key))  # 把query扩展到src_len维度，用广播的方式与key相加，然后计算注意力分数
            scores = scores.squeeze(2).unsqueeze(1)  # scores shape: [batch_size, src_len, 1] -> [batch_size, 1, src_len]

        elif self.type == 'Vaswani':
            scores = torch.bmm(proj_query.unsqueeze(1), proj_key.permute(0, 2, 1)) / math.sqrt(self.hid_dim)  # scores shape: [batch_size, 1, src_len] 
        
        else:
            raise ValueError('Unknown attention type, either Bahdanau or Vaswani.')
        
       
        scores.data.masked_fill_(mask == 0, -float('inf'))  # mask遮蔽掉padding位置
        
        alphas = F.softmax(scores, dim = -1)
        self.alphas = alphas
        
        attn = torch.bmm(self.dropout(alphas), value).squeeze(1)
        # attn shape: [batch_size, 1, hid_dim * 2] -> [batch_size, hid_dim * 2], 只有一次查询，所以可以去掉第二维作为查询结果
        
        return attn, alphas

class Encoder(nn.Module):
    '''
    Encoder of Seq2Seq, consisting of a BiLSTM.
    '''
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        if n_layers == 1:    # 当num_layers为1时，dropout不起作用，不这样写训练时会有警告提示
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True, batch_first=True, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        '''
        Args:
            src: [batch_size, src_len], sentences to be encoded.
        Output:
            outputs: [batch_size, src_len, hid_dim * 2], output of the BiLSTM.
            cell: [n_layers, batch size, hid_dim], average of the final forward and reverse cell state.
        '''
        
        embedded = self.dropout(self.embedding(src))
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        cell_out = []
        for i in range(self.n_layers):
            cell_out.append((cell[i * 2] + cell[i * 2 + 1]) / 2)  # BiLSTM的前向和后向cell state取平均作为上下文表示
        cell_out = torch.stack(cell_out, dim = 0)
        
        hidden_out = []
        for i in range(self.n_layers):
            hidden_out.append((hidden[i * 2] + hidden[i * 2 + 1]) / 2)  # BiLSTM的前向和后向hidden state取平均作为上下文表示
        hidden_out = torch.stack(hidden_out, dim = 0)


        return outputs, hidden_out, cell_out
    
class Decoder(nn.Module):
    '''
    Decoder of Seq2Seq, consisting of a LSTM.
    '''
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        if n_layers == 1:  # 与Encoder同理
            self.rnn = nn.LSTM(emb_dim + hid_dim * 2, hid_dim, n_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(emb_dim + hid_dim * 2, hid_dim, n_layers, batch_first=True, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)

        
    def forward(self, input, attn, hidden, cell):
        '''
        Args:
            input: [batch_size], previous decode result/groud truth (if tearcher force).
            attn: [batch_size, hid_dim * 2], context from the attention mechanism.
            hidden: [n_layers, batch_size, hid_dim], previous hidden state.
            cell: [n_layers, batch_size, hid_dim], previous cell state.
        Output:
            prediction: [batch_size, output_dim], prediction of the current decoding step.
            hidden: [n_layers, batch_size, hid_dim], hidden state of the current decoding step.
            cell: [n_layers, batch_size, hid_dim], cell state of the current decoding step.
        '''
        
        input_emb = self.dropout(self.embedding(input))


        concat_input = torch.cat((input_emb, attn), dim = 1).unsqueeze(1)  
        output, (hidden, cell) = self.rnn(concat_input, (hidden, cell))  
        
        prediction = self.fc_out(output.squeeze(1))
        
        
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    '''
    Seq2Seq model with attention mechanism for Named Entity Recognition, converting a sequence of characters to a sequence of BIO pattern tags.
    '''
    def __init__(self, args, encoder, decoder, attention, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.attention = attention
        self.bm = args.bm

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
        assert encoder.n_layers == decoder.n_layers, \
           "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        '''
        Args:
            src: [batch_size, src_len], sentences to be encoded to BIO sequences.
            trg: [batch_size, trg_len], ground truth BIO sequences, used for training.
            teacher_forcing_ratio: probability to use teacher forcing, 0 to turn off.
        Output:
            outputs: [batch_size, trg_len, output_dim] for training and infering (if bm == 0), indicates the probability of each BIO tag.
                     [trg_len] for infering (if bm != 0), indicates the predicted  BIO sequence.
        Notice:
            trg_len == src_len, both indicate the length of the sequence. 
        '''
        
        batch_size = trg.shape[0] if trg is not None else 1
        src_len = src.shape[1]
        trg_vocab_size = self.decoder.output_dim
        

        outputs_e, hidden, cell = self.encoder(src.long())
        # hidden = torch.zeros((n_layer, batch_size, trg_hid_dim)).to(device)
        
        attn, _ = self.attention(hidden[-1], outputs_e, outputs_e, src.unsqueeze(1))  # key and value are same for Bahdanau attention
        


        if not self.training and self.bm:  # 使用beam_search进行infer
            input = torch.LongTensor([SOS_TOKEN]).to(self.device)
            inputs = [(input, hidden, cell, attn)]
            beam_outputs = []  # bm个元素，存放每条束的outputs
            beam_scores = torch.zeros(batch_size, self.bm).to(self.device)

            for t in range(1, src_len):
                beam_state = []  # 存放一个时间步中每个beam的hidden和cell
                tmp_scores = []
                for i, (input_t, hidden_t, cell_t, attn_t) in enumerate(inputs):
                    output, tmp_hidden, tmp_cell = self.decoder(input_t.long(), attn_t, hidden_t, cell_t)
                    tmp_attn, _ = self.attention(tmp_hidden[-1], outputs_e, outputs_e, src.unsqueeze(1))
                    beam_state.append((input_t, tmp_hidden, tmp_cell, tmp_attn))  # 存放当前时间步各束来源的状态，包括来源(前一个词), hidden, cell和attn
                    tmp_scores.append(beam_scores[:, i].unsqueeze(1) + F.softmax(output, dim = 1))  # bm个元素，每个元素: [N, D]
                
                tmp_scores = torch.cat(tmp_scores, dim = 1)  # shape: [N, D * bm]
                beam_scores, topk_indices = tmp_scores.topk(self.bm, dim = 1)

                inputs = []
                new_beam_outputs = []
                for i, index in enumerate(topk_indices.t()):
                    if len(beam_outputs) < self.bm:  # 针对t = 1的情况
                        beam_outputs.append([beam_scores[0][i], [index % trg_vocab_size]])
                    else:
                        for _, path in beam_outputs:
                            if path[-1] == beam_state[index // trg_vocab_size][0]:
                                new_beam_outputs.append([beam_scores[0][i], path + [index % trg_vocab_size]])
                                break
                        
                    

                    inputs.append((index % trg_vocab_size, 
                                   beam_state[index // trg_vocab_size][1], 
                                   beam_state[index // trg_vocab_size][2], 
                                   beam_state[index // trg_vocab_size][3]))
                    

                beam_outputs.extend(new_beam_outputs)
                beam_outputs.sort(key = lambda x: (-len(x[1]), -x[0]))
                beam_outputs = beam_outputs[:self.bm]


            outputs = beam_outputs[0][-1]

                   
        else:  # train或者evaluate时
            input = trg[:,0] if trg is not None else torch.LongTensor([SOS_TOKEN]).repeat(batch_size).to(device) # SOS token
            outputs = torch.zeros(batch_size, src_len, trg_vocab_size).to(self.device)

            for t in range(1, src_len):
                
                output, hidden, cell = self.decoder(input.long(), attn, hidden, cell)  # output shape: [batch_size, output_dim]
                attn, _ = self.attention(hidden[-1], outputs_e, outputs_e, src.unsqueeze(1))
                outputs[:,t] = output

                teacher_force = np.random.random() < teacher_forcing_ratio
                
                top1 = output.argmax(1) 
                
                input = trg[:,t] if teacher_force else top1  # teacher force or not
        
        return outputs
