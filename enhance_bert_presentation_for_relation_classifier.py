'''
    训练阶段函数,主函数
'''
from all_super_params import test_data_path,train_data_path,relation2id_dict_path,ratio_tr_d_te \
    ,english_bert_base_dir_,bert_config_txt_path,learning_rate,DROPOUT,EPOCHES_NUM,BATCH_SIZE
from c_bert_model import C_Bert_for_RE_model
from pytorch_pretrained_bert import BertModel,BertConfig
from fasNLP_dataloader import Semeval_task_8_corpus_Pipe
from fastNLP.core import AccuracyMetric,CrossEntropyLoss,Adam
import json
from fastNLP.core.const import Const as C
from fastNLP import Trainer

config =BertConfig.from_json_file(bert_config_txt_path)
bert_word_dims =config.hidden_size



bert_model_real =BertModel.from_pretrained(pretrained_model_name_or_path=english_bert_base_dir_)

if __name__ == '__main__':
    print('##'*10)
    data_pipe = Semeval_task_8_corpus_Pipe().process_from_file({'train': train_data_path, 'test': test_data_path})
    print('##'*10)
    with open(relation2id_dict_path, mode='rt', encoding='utf-8') as dict_inp:
        relation2id_dict = json.load(dict_inp)
    c_bert_model =C_Bert_for_RE_model(bert_dims=bert_word_dims,n_class=len(relation2id_dict),bert_model=bert_model_real,dropout=0.1)
    metric =AccuracyMetric(pred=C.OUTPUT,target=C.TARGET)
    loss =CrossEntropyLoss(pred=C.OUTPUT,target=C.TARGET)
    optimizer =Adam(lr=learning_rate,model_params=c_bert_model.parameters())
    trainer =Trainer(train_data=data_pipe.get_dataset('train'),dev_data=data_pipe.get_dataset('test'),model=c_bert_model,optimizer=optimizer, \
                     metrics=metric,batch_size=BATCH_SIZE,n_epochs=EPOCHES_NUM,save_path='c_bert_model_save_dir',loss=loss \
                     )
    result =trainer.train(load_best_model=True)
    print('训练完毕，全局信息:')
    for i,j in result.items():
        print(i,':',j)
