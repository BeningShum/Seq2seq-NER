import json
from collections import Counter

from config import *

from utils import *

class Metrics(object):
    '''
    用于评价模型，计算TokenWise和EntityWise的精确率，召回率，F1分数
    '''

    def __init__(self, golden_tags, predict_tags, remove_O = False):

        self.predict_origin = predict_tags
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:  # 将O标记移除，只关心实体标记
            self._remove_Otags()


        self.cal_tag()  # 计算token级别的精确率、召回率、F1分数
        self.cal_entity()  # 计算实体级别的精确率、召回率、F1分数
   

    def _remove_Otags(self):

        length = len(self.golden_tags)
        golden_tags = []
        predict_tags = []
        # print(len(self.golden_tags))
        # print(len(self.predict_tags))
        for i in range(length):
            if(self.golden_tags[i] != 'O'): # and self.predict_tags[i] == 'O'
                golden_tags.append(self.golden_tags[i])
                predict_tags.append(self.predict_tags[i])
        
        print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
            length,
            len(self.golden_tags)-len(golden_tags),
            (len(self.golden_tags)-len(golden_tags) )/ length * 100
        ))
        self.golden_tags = golden_tags
        self.predict_tags = predict_tags

    def cal_tag(self):
        self.correct_tags_number = self._count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)
        self.tag_set = self.golden_tags_counter.keys()
        
        self.num_tags = len(self.golden_tags)  # 真实BIO标注中的标记总数

        self.tag_p = self._cal_precision(self.tag_set, self.correct_tags_number, self.predict_tags_counter)  # 计算token级别精确率
        self.tag_r = self._cal_recall(self.tag_set, self.correct_tags_number, self.golden_tags_counter)  # 计算token级别召回率
        self.tag_f1 = self._cal_f1(self.tag_set, self.tag_p, self.tag_r)  # 计算token级别F1分数

    def cal_entity(self):
        self.correct_entity_number = self._count_correct_entities()
        self.golden_entity_counter, self.predict_entity_counter = self._get_entities()
        self.entity_set = self.golden_entity_counter.keys()

        self.num_entities = sum(self.golden_entity_counter.values())  # 真实BIO标注中的实体总数

        self.entity_p = self._cal_precision(self.entity_set, self.correct_entity_number, self.predict_entity_counter)  # 计算实体级别精确率
        self.entity_r = self._cal_recall(self.entity_set, self.correct_entity_number, self.golden_entity_counter)  # 计算实体级别召回率
        self.entity_f1 = self._cal_f1(self.entity_set, self.entity_p, self.entity_r)  # 计算实体级别F1分数

    def _count_correct_tags(self):
        """计算每种标签预测正确的个数(对应精确率、召回率计算公式上的tp)，用于后面精确率以及召回率的计算"""
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1

        return correct_dict

    def _get_entities(self):
        entity_golden = {}
        entity_predict = {}
        for i in self.golden_tags:
            if i[0] == 'B':
                if (i[2:] in entity_golden):
                    entity_golden[i[2:]] += 1
                else:
                    entity_golden[i[2:]] = 1

        for i in self.predict_tags:
            if i[0] == 'B':
                if (i[2:] in entity_predict):
                    entity_predict[i[2:]] += 1
                else:
                    entity_predict[i[2:]] = 1
        
        return entity_golden, entity_predict

    def _count_correct_entities(self):
        entity_correct = {}
        start_idx = 0
        end_idx = 0
        for i, tag in enumerate(self.golden_tags):
            if tag[0] == 'B':
                start_idx = i
                end_idx = i + 1
                label = tag[2:]
                while end_idx < len(self.golden_tags) and self.golden_tags[end_idx]  == ('I-' + label):  # 找到实体的结束位置
                    end_idx += 1

                if self.predict_tags[start_idx:end_idx] == self.golden_tags[start_idx:end_idx]:

                    if self.golden_tags[start_idx][2:] in entity_correct:
                        entity_correct[label] += 1
                    else:
                        entity_correct[label] = 1
        
        return entity_correct

    def _cal_precision(self, labels, correct, predict):
        precision_scores = {}

        for label in labels:
            precision_scores[label] = correct.get(label, 0) / predict[label] if predict.get(label, 0) else 0

        return precision_scores

    def _cal_recall(self, labels, correct, golden):
        recall_scores = {}

        for label in labels:
            recall_scores[label] = correct.get(label, 0) / golden[label]
        return recall_scores

    def _cal_f1(self, labels, ps, rs):
        f1_scores = {}
        
        for label in labels:
            p, r = ps[label], rs[label]
            f1_scores[label] = 2 * p * r / (p + r + 1e-10)  # 加上一个特别小的数，防止分母为0
        return f1_scores

    def _cal_micro_PR(self, labels, correct, predict, golden):

        micro_PR = {}


        tp = 0
        tp_fp = 0
        tp_fn = 0
        for label in labels:
            tp += correct.get(label, 0)
            tp_fp += predict.get(label, 0) 
            tp_fn += golden[label]

        micro_PR['precision'] = tp / tp_fp if tp_fp != 0 else 0.
        micro_PR['recall'] = tp / tp_fn if tp_fn != 0 else 0.
        micro_PR['f1_score'] = 2 * micro_PR['precision'] * micro_PR['recall'] / (micro_PR['precision'] + micro_PR['recall'] + 1e-10)

        return micro_PR
    
    def _cal_macro_PR(self, labels, ps, rs):
        macro_PR = {}
        p = 0  # 各个标签的精确率之和
        r = 0  # 各个标签的召回率之和
        for label in labels:
            p += ps[label]
            r += rs[label]

        macro_PR['precision'] = p / len(labels)
        macro_PR['recall'] = r / len(labels)
        macro_PR['f1_score'] = 2 * macro_PR['precision'] * macro_PR['recall'] / (macro_PR['precision'] + macro_PR['recall'] + 1e-10)
        
        return macro_PR

    def _cal_weighted_PR(self, labels, ps, rs, nums, golden):
        weighted_PR = {}
        p = 0
        r = 0
        for label in labels:
            p += ps[label] * golden[label]
            r += rs[label] * golden[label]
        
        p = p / nums
        r = r / nums

        weighted_PR['precision'] = p
        weighted_PR['recall'] = r
        weighted_PR['f1_score'] = 2 * p * r / (p + r + 1e-10)

        return weighted_PR
    

    def report_scores(self):
        '''
        将结果用表格的形式打印出来
        '''
        
        print("--------------------token level ---------------------")
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'
     
        for tag in self.tag_set:  # 打印每个标签的 精确率、召回率、f1分数
            print(row_format.format(
                tag,
                self.tag_p[tag],
                self.tag_r[tag],
                self.tag_f1[tag],
                self.golden_tags_counter[tag]
            ))

        micro_PR = self._cal_micro_PR(self.tag_set, self.correct_tags_number, self.predict_tags_counter, self.golden_tags_counter)
        print(row_format.format(
            'micro_avg',
            micro_PR['precision'],
            micro_PR['recall'],
            micro_PR['f1_score'],
            len(self.golden_tags)
        ))

        macro_PR = self._cal_macro_PR(self.tag_set, self.tag_p, self.tag_r)
        print(row_format.format(
            'macro_avg',
            macro_PR['precision'],
            macro_PR['recall'],
            macro_PR['f1_score'],
            len(self.golden_tags)
        ))

        weighted_PR = self._cal_weighted_PR(self.tag_set, self.tag_p, self.tag_r, self.num_tags, self.golden_tags_counter)
        print(row_format.format(
            'weighted',
            weighted_PR['precision'],
            weighted_PR['recall'],
            weighted_PR['f1_score'],
            len(self.golden_tags)
        ))
        
       

        print("--------------------entity level ---------------------")
        header_format = '{:>9s}  {:>9} {:>9} {:>9} {:>9}'
        header = ['precision', 'recall', 'f1-score', 'support']
        print(header_format.format('', *header))

        row_format = '{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}'

        for tag in self.entity_set:
            print(row_format.format(
                tag,
                self.entity_p[tag],
                self.entity_r[tag],
                self.entity_f1[tag],
                self.golden_entity_counter[tag]
            ))
        
        

        micro_PR = self._cal_micro_PR(self.entity_set, self.correct_entity_number, self.predict_entity_counter, self.golden_entity_counter)
        print(row_format.format(
            'micro_avg',
            micro_PR['precision'],
            micro_PR['recall'],
            micro_PR['f1_score'],
            self.num_entities
        ))

        macro_PR = self._cal_macro_PR(self.entity_set, self.entity_p, self.entity_r)
        print(row_format.format(
            'macro_avg',
            macro_PR['precision'],
            macro_PR['recall'],
            macro_PR['f1_score'],
            self.num_entities
        ))
        
        weighted_PR = self._cal_weighted_PR(self.entity_set, self.entity_p, self.entity_r, self.num_entities, self.golden_entity_counter)
        print(row_format.format(
            'weighted',
            weighted_PR['precision'],
            weighted_PR['recall'],
            weighted_PR['f1_score'],
            self.num_entities
        ))


    def save_ner_results(self, input_test, dir):
        '''
        将测试集的实体识别结果保存在json文件中
        '''
        results = []
        file_path = dir + '/ner_results.json'
        for sen, entities in zip(input_test, parse_entities(input_test, self.predict_origin)):
            result = {}
            if dir.endswith('conll2003'):  # 如果是conll2003数据集(英文)，需要加空格
                result['text'] = ' '.join(sen)
            else:
                result['text'] = ''.join(sen)

            result['entities'] = entities
            results.append(result)

        with open(file_path, "w", encoding = 'utf-8') as f:
            json.dump(results, f, indent = 4, ensure_ascii = False)
