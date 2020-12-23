import json
import os
from utils.dataUtils import make_slot_meta, EXPERIMENT_DOMAINS
import torch
from fix_label import fix_general_label_error
import random
domain2id = {d: i for i, d in enumerate(EXPERIMENT_DOMAINS)}

SLOT_GATE = ["pad","predict", "dontcare", "none"]

slotgate2id = {s: i for i, s in enumerate(SLOT_GATE)}


def load_ontology(main_dir):
    ontology = json.load(open(os.path.join(main_dir, "ontology.json")))
    slot_meta, ontology = make_slot_meta(ontology)
    return slot_meta


conf = json.load(open("conf/config.json"))
SLOT = load_ontology(conf["main_dir"])
slot2id = {s: i for i, s in enumerate(SLOT)}

def load_json_data(filename):
    D = []
    data = json.load(open(filename))

    for dialogue in data:
        content = ""
        for turn in dialogue["dialogue"]:
            domain = turn["domain"]

            if domain not in EXPERIMENT_DOMAINS:
                continue
            turn_dialog_state = fix_general_label_error(turn["belief_state"], False, SLOT)
            content += turn["system_transcript"]
            content += turn["transcript"]


            D.append((content, domain, turn_dialog_state))
    random.shuffle(D)
    # sorted(D,key=lambda x:len(x[0]))
    return D


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1



class DataGenerator():
    def __init__(self, json_path, tokenizer, device):
        self.train_data = load_json_data(json_path)

        self.tokenizer = tokenizer
        self.device = device

    def batchIter(self, batch_size):
        batch_content = []
        batch_domain = []
        batch_gold_state = []

        batch_content_ids = []
        batch_domain_id = []
        batch_slot_pointer_id = [] #[batch,slot]
        batch_slot_gate_id = []  #[batch,max_slot] #涉及的槽,pointer是1
        batch_value_start = [] #[batch,max_predict] #需要预测的槽,gate是predict
        batch_value_end = [] #[batch,max_predict]

        for cnt, (content, domain, turn_dialog_state) in enumerate(self.train_data):

            context_ids = self.tokenizer.encode(content)
            if len(context_ids)>200:
                context_ids=context_ids[len(context_ids)-200:]
                content=self.tokenizer.decode(context_ids)
            domainid = domain2id[domain]
            slot_pointer_id = [0] * len(SLOT)
            gold_state = []
            slot_gate_ids = []
            value_starts = []
            value_ends = []
            last_slot_id=-1
            for slot in SLOT:
                if slot not in turn_dialog_state:
                    continue
                value=turn_dialog_state[slot]
                if slot not in SLOT:
                    continue
                gold_state.append(slot+"-"+value)

                slotgate = "predict"
                if value in SLOT_GATE:
                    slotgate = value



                value_ids = self.tokenizer.encode(value)[1:-1]
                start = search(value_ids, context_ids)

                if slotgate == "predict" and start == -1  :# 不在原文出现
                    continue

                slot_id = slot2id[slot]

                assert slot_id>last_slot_id
                last_slot_id=slot_id

                slot_pointer_id[slot_id] = 1

                slot_gate_id = slotgate2id[slotgate]
                slot_gate_ids.append(slot_gate_id)

                if slotgate=="predict":
                    value_starts.append(start)
                    value_ends.append(start + len(value_ids) - 1)


            batch_content.append(content)
            batch_domain.append(domain)
            batch_gold_state.append(gold_state)
            batch_content_ids.append(context_ids)
            batch_domain_id.append(domainid)
            batch_slot_pointer_id.append(slot_pointer_id)
            batch_slot_gate_id.append(slot_gate_ids)
            batch_value_start.append(value_starts)
            batch_value_end.append(value_ends)

            if len(batch_content) >= batch_size or cnt == len(self.train_data) - 1:
                ret = self.tokenizer.batch_encode_plus(batch_content, pad_to_max_length=True)

                batch_content_ids = ret['input_ids']
                batch_token_type_ids = ret['token_type_ids']
                batch_attention_mask = ret['attention_mask']

                # mask batch_slot_gate_id
                max_slot = 0
                for slot_gate_id in batch_slot_gate_id:
                    max_slot = max(max_slot, len(slot_gate_id))
                for i in range(len(batch_slot_gate_id)):
                    slot_num = len(batch_slot_gate_id[i])
                    batch_slot_gate_id[i] = batch_slot_gate_id[i] + (max_slot - slot_num) * [0]

                # mask batch_value_start , batch_value_end
                max_predict = 0
                for value_start,value_end in zip(batch_value_start,batch_value_end):
                    max_predict = max(max_predict, len(value_start))
                    assert len(value_start)==len(value_end)
                for i in range(len(batch_value_start)):
                    predict_num = len(batch_value_start[i])
                    batch_value_start[i] = batch_value_start[i] + (max_predict - predict_num) * [0]
                    batch_value_end[i] = batch_value_end[i] + (max_predict - predict_num) * [0]

                batch_content_ids = torch.tensor(batch_content_ids).to(self.device)
                batch_token_type_ids = torch.tensor(batch_token_type_ids).to(self.device)
                batch_attention_mask = torch.tensor(batch_attention_mask).to(self.device)
                batch_domain_id = torch.tensor(batch_domain_id).to(self.device)
                batch_slot_pointer_id = torch.tensor(batch_slot_pointer_id).to(self.device)
                batch_slot_gate_id = torch.tensor(batch_slot_gate_id).to(self.device)
                batch_value_start = torch.tensor(batch_value_start).to(self.device)
                batch_value_end = torch.tensor(batch_value_end).to(self.device)

                yield {
                    "batch_content_ids": batch_content_ids,
                    "batch_token_type_ids": batch_token_type_ids,
                    "batch_attention_mask": batch_attention_mask,
                    "batch_domain_id": batch_domain_id,
                    "batch_slot_pointer_id": batch_slot_pointer_id,
                    "batch_slot_gate_id": batch_slot_gate_id,
                    "batch_value_start": batch_value_start,
                    "batch_value_end": batch_value_end,
                    "batch_gold_state": batch_gold_state
                }

                batch_content = []
                batch_domain = []
                batch_gold_state = []
                batch_content_ids = []
                batch_domain_id = []
                batch_slot_pointer_id = []
                batch_slot_gate_id = []
                batch_value_start = []
                batch_value_end = []
