import torch
import torch.nn as nn
from utils.dataUtils import EXPERIMENT_DOMAINS
from data import SLOT, SLOT_GATE, slotgate2id


class config:
    def __init__(self):
        self.hidden_size = 768 #128  # 768
        self.dropout = 0.1


class BertDST(nn.Module):
    def __init__(self, tokenizer, encode_model, device):
        super(BertDST, self).__init__()

        # 1. bert encode
        self.tokenizer = tokenizer
        self.encode_model = encode_model
        self.device = device

        args = config()
        self.args = config()

        self.domain_linear = nn.Linear(args.hidden_size, len(EXPERIMENT_DOMAINS))

        self.slot_linear = nn.Linear(args.hidden_size, len(SLOT))

        self.slot_gate_linears = nn.ModuleList(
            [nn.Linear(args.hidden_size, self.args.hidden_size) for i in range(len(SLOT))])

        self.slot_gate_linear = nn.Linear(args.hidden_size, len(SLOT_GATE))

        self.start_linears = nn.ModuleList([nn.Linear(args.hidden_size, 1) for i in range(len(SLOT))])
        self.end_linears = nn.ModuleList([nn.Linear(args.hidden_size, 1) for i in range(len(SLOT))])

    def forward(self, batch_content_ids, batch_token_type_ids, batch_attention_mask, slot_pointer=None, slot_gate=None):
        output = self.encode_model(batch_content_ids, attention_mask=batch_attention_mask,
                                   token_type_ids=batch_token_type_ids)
        H = output[0]
        # H [batch,len,dim]
        cls_output = output[1]
        # cls_output [batch,dim]

        # 领域 ok
        domain_score = self.domain_linear(cls_output)
        # domain_score [batch,domain]

        # 涉及的槽 ok
        slot_score = self.slot_linear(cls_output)
        # slot_score [batch,slot]
        sigmoid = nn.Sigmoid()
        slot_pointer_prob = sigmoid(slot_score)
        # slot_pointer_prob [batch,slot]





        # 槽门,只用涉及到的槽

        if slot_pointer is None:
            slot_pointer = (slot_pointer_prob > 0.5)
            # slot_pointer [batch,slot]

        cls_hidden = cls_output.unsqueeze(1).expand(-1, len(SLOT), -1)
        # cls_hidden [batch,slot,dim]
        slot_hidden = torch.zeros(cls_hidden.size(),device=self.device)
        # slot_hidden [batch,slot,dim]


        for i in range(len(SLOT)):
            slot_hidden[:, i, :] = self.slot_gate_linears[i](cls_hidden[:, i, ])

        max_slot_num = slot_pointer.sum(dim=-1).max().item()
        selected_slot_gate_hidden = []
        # selected_slot_gate_hidden [batch,max_slot_num,dim]
        batch_size = slot_pointer.size(0)
        for i in range(batch_size):

            # row_slot_gate_hidden [max_slot_num,dim]

            if slot_pointer[i].sum().item() > 0:
                row_slot_gate_hidden = torch.masked_select(slot_hidden[i], #[slot,dim]
                                                           slot_pointer[i].eq(1).unsqueeze(1)).view(-1,self.args.hidden_size)

                assert len(row_slot_gate_hidden)==slot_pointer[i].sum().item()
                gap = max_slot_num - len(row_slot_gate_hidden)
                if gap > 0:
                    zeros = torch.zeros((gap, self.args.hidden_size), device=self.device)
                    row_slot_gate_hidden = torch.cat([row_slot_gate_hidden, zeros], 0)
            else:
                row_slot_gate_hidden = torch.zeros((max_slot_num, self.args.hidden_size), device=self.device)
            selected_slot_gate_hidden.append(row_slot_gate_hidden.unsqueeze(0)) #[1,max_slot,dim]
        selected_slot_gate_hidden = torch.cat(selected_slot_gate_hidden, 0)
        # selected_slot_gate_hidden [batch,max_slot_num,dim]
        slot_gate_score = self.slot_gate_linear(selected_slot_gate_hidden)
        # slot_gate_score [batch, max_slot_num, slot_gate]
        slot_gate_prob = nn.Softmax(dim=-1)(slot_gate_score)
        # slot_gate_prob [batch,max_slot_num,slot_gate]

        if slot_gate is None:
            slot_gate = slot_gate_score.max(-1)[-1]
            # slot_gate [batch, max_slot_num]







        # 槽门是predict的， 求跨度 ok
        max_predict = slot_gate.eq(slotgate2id["predict"]).sum(-1).max().item()

        # H [batch,len,dim]
        max_content_len = H.size(1)


        H_hidden=H.unsqueeze(1).expand(-1,len(SLOT),-1,-1)
        # H_hidden [batch,slot,len,dim]
        start_score=torch.zeros((batch_size,len(SLOT),max_content_len),device=self.device)
        # start_score [batch,slot,len]
        end_score = torch.zeros((batch_size, len(SLOT), max_content_len),device=self.device)
        # end_score [batch,slot,len]
        for i in range(len(SLOT)):
            start_score[:, i,: ] = self.start_linears[i](H_hidden[:, i, :]).squeeze(-1)
            end_score[:, i, :] = self.end_linears[i](H_hidden[:, i, :]).squeeze(-1)

        selected_start_score=torch.zeros((batch_size, max_predict, max_content_len),device=self.device)
        # selected_start_score [batch,max_predict,len]
        selected_end_score=torch.zeros((batch_size, max_predict, max_content_len),device=self.device)
        # selected_end_score [batch,max_predict,len]


        for i in range(batch_size):
            j=0 #pointer
            k=0 #predict
            for slot_index in range(len(SLOT)):
                if slot_pointer[i][slot_index]==1:
                    if slot_gate[i][j]==slotgate2id["predict"]:
                        selected_start_score[i][k] = start_score[i][slot_index] # start_score end_score 没有压缩,所以是slot_index
                        selected_end_score[i][k] = end_score[i][slot_index]

                        k+=1
                    j+=1


        start_prob = nn.Softmax(dim=-1)(selected_start_score)
        # start_prob [batch,max_predict,len]

        end_prob = nn.Softmax(dim=-1)(selected_end_score)
        # end_prob [batch,max_predict,len]

        return domain_score, slot_pointer_prob, slot_gate_prob, slot_pointer, slot_gate, start_prob, end_prob
