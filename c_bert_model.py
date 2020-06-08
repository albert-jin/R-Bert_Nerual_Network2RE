from pytorch_pretrained_bert import BertModel
from fastNLP.core.const import Const as C
print('CONST常量有:', C.TARGET,C.INPUT,C.LOSS,C.INPUT_LEN,C.RAW_WORD)
import torch as t
class Lambda(t.nn.Module):
    def __init__(self,lambda_):
        super().__init__()
        self.func =lambda_
    def forward(self,x,eb,ee):
        return self.func(x,eb,ee)

class C_Bert_for_RE_model(t.nn.Module):
    def __init__(self,bert_model:BertModel,bert_dims,n_class,dropout=0):
        super().__init__()
        self.bert_model =bert_model
        self.e1_fc =t.nn.Linear(in_features=bert_dims,out_features=bert_dims) # 该full_connect模块默认使用偏置
        self.e2_fc =t.nn.Linear(in_features=bert_dims,out_features=bert_dims)
        self.sent_fc =t.nn.Linear(in_features=bert_dims,out_features=bert_dims)
        self.lambda_ =Lambda(self.func_)
        self.activation =t.nn.ReLU()
        self.dropout =t.nn.Dropout(p=dropout)
        self.fc_for_softmax =t.nn.Linear(in_features=self.sent_fc.out_features*3,out_features=n_class)
        self.cuda()
        self.apply(self.init_weight)
    def init_weight(self,sub_module):
        if isinstance(sub_module,t.nn.Linear) and sub_module in self.children():
            t.nn.init.kaiming_normal_(sub_module.weight,nonlinearity='relu')
    def func_(self,x,eb,ee):
        ''' x:batch_size * sequence_len * embeddings , e1,e2,e3,e4 :batch_size * 1
            :returns ： batch_size *embeddings
        '''
        eb =eb.cpu().numpy()
        ee =ee.cpu().numpy()
        list_ =[]
        for idx,(i,j) in enumerate(zip(eb,ee)):
            list_.append(t.mean(x[idx,i+1:j,:],dim=0,keepdim=False).unsqueeze(dim=0))
        return t.cat(list_)

    def forward(self,words_bert_ids,e1b,e2b,e1e,e2e):
        bert_sent_output,bert_pool_output =self.bert_model(words_bert_ids,output_all_encoded_layers =False)
        # batch_size,sequence_len,embeddings & batch_size,embeddings

        e1 =self.lambda_(bert_sent_output,e1b,e1e) #bert_sent_output[:,e1b+1:e1e,:]
        e2 =self.lambda_(bert_sent_output,e2b,e2e) #bert_sent_output[:,e2b+1:e2e,:]
        # mean_e1 =t.mean(e1,dim=1,keepdim=False) # batch_size,embeddings
        # mean_e2 =t.mean(e2,dim=1,keepdim=False) # batch_size,embeddings
        mean_e1 =e1
        mean_e2 =e2
        e1_final =self.activation(self.e1_fc(mean_e1)) # batch_size,out_features
        e1_final =self.dropout(e1_final)
        e2_final =self.activation(self.e2_fc(mean_e2)) # batch_size,out_features
        e2_final =self.dropout(e2_final)
        pool_final =self.activation(self.sent_fc(bert_pool_output)) # batch_size,out_features
        pool_final =self.dropout(pool_final)
        all_final =t.cat([e1_final,e2_final,pool_final],dim=1) # batch_size,out_features*3
        all_final =self.fc_for_softmax(all_final) # batch_size,num_class
        all_final =self.dropout(all_final)
        all_final =t.softmax(all_final,dim=1)
        return {C.OUTPUT:all_final}

    def predict(self,words_bert_ids,e1b,e2b,e1e,e2e):
        output =self(words_bert_ids,e1b,e2b,e1e,e2e)
        _,predict =output[C.OUTPUT].max(dim =1)
        return {C.OUTPUT:predict}







