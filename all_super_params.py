train_data_path = 'SemEval2010-Task8/train/FULL_TRAIN.txt'
test_data_path = 'SemEval2010-Task8/test/FULL_TEST.txt'
relation2id_dict_path ='relation2id_dict.json'

ratio_tr_d_te =(8,1,1)   #分割semeval 为train/test/val的比例

english_bert_base_dir_ =r'C:\Users\24839\Desktop\科研\bert预训练模型\pytorch_bert_base_english'

bert_vocab_txt =r'C:\Users\24839\Desktop\科研\bert预训练模型\pytorch_bert_base_english\bert-base-uncased-vocab.txt'

bert_config_txt_path =r'C:\Users\24839\Desktop\科研\bert预训练模型\pytorch_bert_base_english\bert_config.json'

BATCH_SIZE =16

MAX_SEQUENCE_LENGTH =200  #作者论文是128

first_entity_tag ='$'
second_entity_tag ='#'

learning_rate =2e-5

EPOCHES_NUM =5

DROPOUT =0.1
