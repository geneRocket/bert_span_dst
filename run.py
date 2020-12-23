from model import BertDST
from data import DataGenerator,slotgate2id,SLOT
import torch
from torch import nn, optim
from transformers import *
from utils.eval_utils import compute_prf, compute_acc
import time
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_batch_size=10
eval_batch_size=10
learning_rate=5e-5
epoch=30

pretrained = 'bert-base-uncased'
# pretrained = 'distilbert-base-uncased'
#pretrained = 'mrm8488/bert-tiny-5-finetuned-squadv2'
tokenizer = BertTokenizer.from_pretrained(pretrained)


def get_batch_predict_answer(token_id, p1, p2, max_a_len=16):
    batch_size, c_len = p1.size()
    ls = nn.LogSoftmax(dim=1)
    mask_pos = (torch.ones(c_len, c_len).to(device) * float('-inf')).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
    mask_len = (torch.ones(c_len, c_len).to(device) * float('-inf')).triu(max_a_len).unsqueeze(0).expand(batch_size, -1,
                                                                                                         -1)

    score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask_pos + mask_len
    score, s_idx = score.max(dim=1)
    score, e_idx = score.max(dim=1)
    s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze(-1)
    answer_list=[]
    for i in range(batch_size):
        answer_list.append(tokenizer.decode(token_id[i][s_idx[i]:e_idx[i] + 1]))
    return  answer_list

def get_state(batch_content_ids,batch_slot_pointer_id,batch_slot_gate_id,start_prob,end_prob):
    states_list=[] # 二维
    (batch, max_predict, _) = start_prob.size()

    predict_batch_index = []
    predict_batch_slots = []
    predict_batch_content_ids = []
    predict_batch_start_prob = []
    predict_batch_end_prob = []

    for i in range(batch):
        states=[]
        j = 0 # 涉及的槽
        k = 0 # 需要predict的槽


        for slot_index in range(len(SLOT)):
            if batch_slot_pointer_id[i][slot_index] == 1:
                slot = SLOT[slot_index]
                if batch_slot_gate_id[i][j] == slotgate2id["predict"]:

                    predict_batch_index.append(i)
                    predict_batch_slots.append(slot)
                    predict_batch_content_ids.append(batch_content_ids[i])
                    predict_batch_start_prob.append(start_prob[i][k].unsqueeze(0))
                    predict_batch_end_prob.append(end_prob[i][k].unsqueeze(0))


                    k += 1
                elif batch_slot_gate_id[i][j] == slotgate2id["dontcare"]:
                    states.append(slot + "-" + "dontcare")
                elif batch_slot_gate_id[i][j] == slotgate2id["none"]:
                    states.append(slot + "-" + "none")
                j += 1


        states_list.append(states)

    if len(predict_batch_content_ids) > 0:
        predict_batch_start_prob=torch.cat(predict_batch_start_prob,0)
        predict_batch_end_prob = torch.cat(predict_batch_end_prob, 0)
        predict_batch_values = get_batch_predict_answer(predict_batch_content_ids, predict_batch_start_prob,predict_batch_end_prob)
        for i in range(len(predict_batch_content_ids)):
            states_list[predict_batch_index[i]].append(predict_batch_slots[i] + "-" + predict_batch_values[i])

    return states_list

def masked_cross_entropy(logits, target, pad_idx):
    # logits [batch,slot,gate]
    # target [batch,slot]
    mask = target.ne(pad_idx) #有用
    # mask [batch,slot]
    logits_flat = logits.view(-1, logits.size(-1))
    # logits_flat [batch*slot,gate]
    log_probs_flat = -torch.log(logits_flat)
    # log_probs_flat [batch*slot,gate]
    target_flat = target.view(-1, 1)
    # target_flat [batch*slot,1]
    losses_flat = torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask
    loss = losses.sum() / (mask.sum().float())
    return loss

def train(infile,test_file):

    encode_model = BertModel.from_pretrained(pretrained)
    model=BertDST(tokenizer, encode_model, device)
    model=model.to(device)
    data_gen=DataGenerator(infile,tokenizer,device)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    crossEntropyLoss = nn.CrossEntropyLoss()
    mseLoss=torch.nn.MSELoss(reduce=True, size_average=True)
    bceLoss=nn.BCELoss()
    model.train()


    for e in range(epoch):
        batch_cnt=0
        for batch in data_gen.batchIter(train_batch_size):
            batch_cnt+=1

            batch_content_ids= batch["batch_content_ids"]
            batch_token_type_ids= batch["batch_token_type_ids"]
            batch_attention_mask= batch["batch_attention_mask"]
            batch_domain_id= batch["batch_domain_id"]
            # batch_domain_id [batch,domain]
            batch_slot_pointer_id= batch["batch_slot_pointer_id"]
            # batch_slot_pointer_id [batch,slot]
            batch_slot_gate_id= batch["batch_slot_gate_id"]
            # batch_slot_gate_id [batch,max_slot]
            batch_value_start= batch["batch_value_start"]
            # batch_value_start [batch,max_predict]
            batch_value_end= batch["batch_value_end"]
            # batch_value_start [batch,max_predict]
            batch_gold_state=batch["batch_gold_state"]

            domain_score, slot_pointer_prob, slot_gate_prob, slot_pointer, slot_gate,start_prob, end_prob=model(batch_content_ids,batch_token_type_ids,batch_attention_mask,batch_slot_pointer_id,batch_slot_gate_id)
            optimizer.zero_grad()

            batch_loss = crossEntropyLoss(domain_score, batch_domain_id)
            #batch_loss+=mseLoss(slot_pointer_prob,batch_slot_pointer_id.float())
            batch_loss += bceLoss(slot_pointer_prob, batch_slot_pointer_id.float())



            if(batch_slot_gate_id.size(1)>0):
                slot_gate_loss = masked_cross_entropy(slot_gate_prob,batch_slot_gate_id,slotgate2id["pad"])
                batch_loss+=slot_gate_loss

            if(start_prob.size(1)>0):
                start_loss = masked_cross_entropy(start_prob,batch_value_start,0)
                end_loss = masked_cross_entropy(end_prob, batch_value_end,0)
                batch_loss+=start_loss
                batch_loss+=end_loss

                a=get_state(batch_content_ids, batch_slot_pointer_id, batch_slot_gate_id, start_prob, end_prob)
                b=batch_gold_state
                a=[sorted(x) for x in a]
                b=[sorted(x) for x in b]
                print(a)
                print(b)
                print(e,batch_cnt*train_batch_size,batch_loss.item())
                print("=========================================")




            batch_loss.backward()
            optimizer.step()

        evaluation(test_file, tokenizer, model, device, e)


        torch.save(model,'model_save')

def evaluation(infile,tokenizer,model,device,epoch):

    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    len_test_data=0

    model.eval()
    data_gen = DataGenerator(infile, tokenizer, device)


    wall_times = []

    for batch in data_gen.batchIter(eval_batch_size):

        batch_content_ids = batch["batch_content_ids"]
        batch_token_type_ids = batch["batch_token_type_ids"]
        batch_attention_mask = batch["batch_attention_mask"]
        batch_gold_state = batch["batch_gold_state"]
        start = time.perf_counter()

        with torch.no_grad():
            domain_score, slot_pointer_prob, slot_gate_prob, slot_pointer, slot_gate,start_prob, end_prob = model(batch_content_ids,
                                                                                          batch_token_type_ids,
                                                                                          batch_attention_mask
                                                                                          )

            state_list=get_state(batch_content_ids, slot_pointer, slot_gate, start_prob, end_prob)

        end = time.perf_counter()
        wall_times.append(end - start)


        for pred_state,gold_state in zip(state_list,batch_gold_state):

            if set(pred_state) == set(gold_state):
                joint_acc += 1
            len_test_data+=1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(gold_state), set(pred_state), SLOT)
            slot_turn_acc += temp_acc

            # Compute prediction F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(gold_state, pred_state)
            slot_F1_pred += temp_f1
            slot_F1_count += count






    joint_acc_score = joint_acc / len_test_data
    turn_acc_score = slot_turn_acc / len_test_data
    slot_F1_score = slot_F1_pred / slot_F1_count
    latency = np.mean(wall_times) * 1000


    print("------------------------------")
    print("Epoch %d joint accuracy : " % epoch, joint_acc_score)
    print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
    print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
    print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")


    scores = {'epoch': epoch, 'joint_acc': joint_acc_score,
              'slot_acc': turn_acc_score, 'slot_f1': slot_F1_score}
    return scores



train("TARGET_PATH/train_dials.json","TARGET_PATH/test_dials.json")


