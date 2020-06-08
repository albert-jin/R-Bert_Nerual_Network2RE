'''
    train_data_path : 'SemEval2010-Task8/train/FULL_TRAIN.txt'

    data_visualization :
8001    "The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
Message-Topic(e1,e2)
Comment: Assuming an audit = an audit document.
<空行分隔每个训练样本>

    test_data+_path : 'SemEval2010-Task8/test/FULL_TEST.txt'

    data_visualization :
1	"The system as described above has its greatest application in an arrayed <e1>configuration</e1> of antenna <e2>elements</e2>."
Component-Whole(e2,e1)
Comment: Not a collection: there is structure here, organisation.
<空行分隔每个训练样本>
'''

'''
    为了复现源论文，实验参数/工具保持一致：
    1. bert:英文版，uncased，base
    2. 添加的实体辨别符号为 $以及#
    3. 实验介绍的超参保持一致
    4. 原文并未进行关系指向的任务，因此一共9+1十种关系
'''

from fastNLP.io import Pipe,Loader,DataBundle
from numpy import random
from fastNLP.core import Instance,DataSet
import warnings
from  typing import Dict,Union
from pathlib import  Path
import os
import json
from all_super_params import test_data_path,train_data_path,relation2id_dict_path,ratio_tr_d_te, \
    MAX_SEQUENCE_LENGTH ,bert_vocab_txt,first_entity_tag,second_entity_tag  # ,english_bert_base_dir_
from pytorch_pretrained_bert import BertTokenizer

class Semeval_task_8_corpus_loader(Loader):
    def __init__(self):
        super().__init__()
        pass
    def _load(self, path: str) -> DataSet:
        '''
        :param path: str 读取的数据集文件，train&test数据集格式一致，可重复使用本function
        :return: DayaSet,load本文件并返回的数据集数目

        :tips: 由于RE训练集中必定含有所有类型关系的标签，
        '''
        inp =open(path,mode='rt',encoding='utf-8')
        ds =DataSet()
        relation_type2relation_id_dict = dict()
        if os.path.exists(os.path.abspath(relation2id_dict_path)):
            with Path(relation2id_dict_path).open(mode='rt',encoding='utf-8') as dict_inp:
                relation_type2relation_id_dict =json.load(dict_inp)
        idx,sentence =inp.readline().strip().split('\t')[:2]
        relation_type =inp.readline().strip().split('(')[0]
        comments =inp.readline().strip()
        blank_row =inp.readline()
        while True:
            if relation_type not in relation_type2relation_id_dict:
                relation_type2relation_id_dict[relation_type] = len(relation_type2relation_id_dict)
                print('出现新relation_type，加入关系映射id字典,键值对:{}'.format({relation_type:relation_type2relation_id_dict[relation_type]}))
            relation_id = relation_type2relation_id_dict[relation_type]
            if sentence.startswith('\"') and sentence.endswith('\"'):
                sentence =sentence[1:-1]
            else:
                raise Exception('Semeval task8 的句子开始&结束必须是:\"')
            if len(sentence.split(' ')) >MAX_SEQUENCE_LENGTH:
                print(path,'中 idx:{}的训练样本长度超过阈值{}，舍弃'.format(idx,MAX_SEQUENCE_LENGTH))
                idx, sentence = inp.readline().strip().split('\t')[:2]
                relation_type = inp.readline().strip().split('(')[0]
                comments = inp.readline().strip()
                continue
            ds.append(Instance(raw_words =sentence,index =idx,target =relation_id,comments =comments))
            sent =inp.readline().strip()
            if '\t' not in sent:
                break
            idx, sentence = sent.strip().split('\t')[:2]
            relation_type = inp.readline().strip().split('(')[0]
            comments = inp.readline().strip()
            blank_row =inp.readline()
        if not os.path.exists(os.path.abspath(relation2id_dict_path)):
            with Path(relation2id_dict_path).open(mode='wt',encoding='utf-8') as outp:
                json.dump(relation_type2relation_id_dict,outp)
        inp.close()
        print(path,'文件中读取的Dataset数据总量为:',len(ds),'个Instance')
        return ds

    def check_loader_paths(self,paths: Union[str, Dict[str, str]]) -> Dict[str, str]:
        # 这一步保证了dict 中有train,1<=len(dict)<=3,key included in {train,test,val},dict的values都是path
        if isinstance(paths,(str,Path)):
            paths =os.path.abspath(os.path.expanduser(paths))
            if os.path.isfile(paths):
                return {'train':paths}
            else:
                raise Exception('读取的路径不是一个文件')
        elif isinstance(paths,dict):
            if 'train' not in paths:
                raise Exception('dict中 键key必须要有“train”')
            for name,path in paths.items():
                if name not in {'train','val','test'}:
                    raise Exception('dict中 键key必须属于train/val/test')
                if not os.path.isfile(os.path.abspath(path)):
                    raise Exception('字典中 name{}->路径:{}不是文件'.format(name,path))
            return paths

    def load(self, paths: Union[str, Dict[str, str]] = None,ratio_tr_d_te:tuple =ratio_tr_d_te) -> DataBundle:
        '''
         :param paths: 为str时，读入训练集合的所有训练样本,并在训练集的基础上按比例8：1：1划分
                    为Dict[str,str]时，通过test，val，train这种键值来pick训练、测试、验证集，（train必须要有）
                    如果没有val、test，从train中划分一定比例充当验证/测试集
        :return:
        '''
        paths =self.check_loader_paths(paths)  #此时的paths是字典{'train':XXX,..}
        datasets ={_:self._load(path=path) for _,path in paths.items()}
        # 对所有数据做shuffle 处理
        for name,ds in datasets.items():
            shuffled_ds =DataSet()
            indices =[_ for _ in range(len(ds))]
            random.shuffle(indices)
            for _ in indices:
                shuffled_ds.append(ds[_])
            datasets[name] =shuffled_ds
        # shuffle 处理结束
        if len(datasets) ==1:
            print('检测到只load train中的dataset，默认8：1：1拆分为train/test/val 三份集合')
            ds =datasets['train']
            train_count =int(len(ds)*(ratio_tr_d_te[0]/sum(ratio_tr_d_te)))
            test_count = int(len(ds)*(ratio_tr_d_te[2]/sum(ratio_tr_d_te)))
            return DataBundle(datasets={'train':ds[:train_count],'val':ds[train_count:-test_count],'test':ds[-test_count:]})
        elif len(datasets) ==3:
            print('检测到train,test,val,不需要从train划分')
            return DataBundle(datasets=datasets)
        elif 'val' not in datasets:
            print('检测到train,test,从train划分出val')
            ds = datasets['train']
            val_count = int(len(ds) * (ratio_tr_d_te[1] / sum(ratio_tr_d_te)))
            return DataBundle(datasets= {'train': ds[:-val_count], 'val': ds[-val_count:], 'test': datasets['test']})
        elif 'test' not in datasets:
            print('检测到train,val,从train划分出test')
            ds = datasets['train']
            test_count = int(len(ds) * (ratio_tr_d_te[2] / sum(ratio_tr_d_te)))
            return DataBundle(datasets={'train': ds[:-test_count], 'val': ds[-test_count:], 'test': datasets['test']})

class Semeval_task_8_corpus_Pipe(Pipe):
    def __init__(self):
        super().__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=bert_vocab_txt)
    def raw_words2words_func(self,raw_words:str)->dict:
        '''
        该函数给Dataset.apply_field_more使用
        :param raw_words:   raw_words 列的sentence 'I can't breathe <e1>entity</e1> has a <e2>peach</e2> , but i like it .'
        :return:字典  --->新建的words_bert_ids 列 [100,2352,435,2342,234,...] 所有实体分割为 $ e1 $ ,# e2 # ；
                      --->并建立四个列 ,代表e1,e2的前后边界
        '''
        if not raw_words.split('<e1>')[0].endswith(' '):
            raw_words =raw_words.split('<e1>')[0]+' <e1>'+raw_words.split('<e1>')[1]
        if not raw_words.split('<e2>')[0].endswith(' '):
            raw_words =raw_words.split('<e2>')[0]+' <e2>'+raw_words.split('<e2>')[1]
        raw_words_list =raw_words.split(' ')
        new_words_list =[]
        four_idx_list =[]
        for word in raw_words_list:
            if word.startswith('<e1>') or word.find('</e1>')!=-1: #满足<e1>开头,或含有</e1>
                if word.startswith('<e1>'):
                    new_words_list.append('$')
                    four_idx_list.append(len(new_words_list)-1)
                    new_words_list.append(word.replace('<e1>','').split('</e1>')[0])
                if word.find('</e1>')!=-1: # 含有</e1>
                    if word.endswith('</e1>'): # </e1>结尾
                        if not word.startswith('<e1>'):
                            new_words_list.append(word.split('</e1>')[0])
                        new_words_list.append('$')
                        four_idx_list.append(len(new_words_list) - 1)
                    else:  # </e1>,结尾
                        if not word.startswith('<e1>'):
                            new_words_list.append(word.split('</e1>')[0])
                        new_words_list.append('$')
                        four_idx_list.append(len(new_words_list) - 1)
                        new_words_list.append(word.split('</e1>')[-1])
            elif word.startswith('<e2>') or word.find('</e2>') != -1:  # 满足<e2>开头,或含有</e2>
                if word.startswith('<e2>'):
                    new_words_list.append('#')
                    four_idx_list.append(len(new_words_list) - 1)
                    new_words_list.append(word.replace('<e2>', '').split('</e2>')[0])
                if word.find('</e2>') != -1:  # 含有</e2>
                    if word.endswith('</e2>'):  # </e2>结尾
                        if not word.startswith('<e2>'):
                            new_words_list.append(word.split('</e2>')[0])
                        new_words_list.append('#')
                        four_idx_list.append(len(new_words_list) - 1)
                    else:  # </e2>,结尾
                        if not word.startswith('<e2>'):
                            new_words_list.append(word.split('</e2>')[0])
                        new_words_list.append('#')
                        four_idx_list.append(len(new_words_list) - 1)
                        new_words_list.append(word.split('</e2>')[-1])
            else:
                new_words_list.append(word)
        assert len(raw_words_list)+6 >=len(new_words_list) >=len(raw_words_list)+4
        assert len(four_idx_list) ==4
        words_bert_ids =self.bert_tokenizer.convert_tokens_to_ids(['[CLS]']+new_words_list+['[SEP]'])
        return {'words_bert_ids':words_bert_ids,'e1b':four_idx_list[0],'e1e':four_idx_list[1],'e2b':four_idx_list[2],'e2e':four_idx_list[3]}
    def process(self, data_bundle: DataBundle) -> DataBundle:
        '''
        :param data_bundle:  databundler里的dataset列为 raw_words|index|target|comment ->用‘raw_words’增加 words_bert_ids|e1e|e1e|e2b|e2e ,还需设置is_input,is_target列属性
        :return: 对databundle中的dataset进行扩展
        '''
        for name,dataset in data_bundle.datasets.items():
            dataset.apply_field_more(func=self.raw_words2words_func,field_name='raw_words',modify_fields=True)
        data_bundle.set_input('words_bert_ids','e1b','e1e','e2b','e2e')
        data_bundle.set_target('target')
        return data_bundle
    def process_from_file(self, paths) -> DataBundle:
        data_bundler =Semeval_task_8_corpus_loader().load(paths=paths,ratio_tr_d_te=ratio_tr_d_te)
        return self.process(data_bundle=data_bundler)

if __name__ == '__main__':
    # 该main function 用来测试dataloader的功能
    data_pipe= Semeval_task_8_corpus_Pipe().process_from_file({'train':train_data_path,'test':test_data_path})
    print(data_pipe.get_dataset('train').print_field_meta())
    print({len(data_pipe.get_dataset(name))for name in ['train','val','test'] })
    print(data_pipe.get_dataset('train')[:2])










