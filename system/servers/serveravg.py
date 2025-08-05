import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
from peft import PeftModel

import copy

from peft import (
    set_peft_model_state_dict,
)
import torch
import os
from torch.nn.functional import normalize
from torch.nn import ZeroPad2d

from peft import set_peft_model_state_dict, get_peft_model, LoraConfig, set_peft_model_state_dict
import torch.nn as nn
from flcore.trainmodel.models import BaseHeadSplit 


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.load_model_1(model_path= "./xxxx.pt")

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        self.Budget = []

        self.best_acc = 0.0
        self.best_round = -1


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                # self.load_model()
                # 只保存最佳accuracy模型
                # self.evaluate()
                current_acc = self.evaluate()
                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    self.best_round = i
                    print(f"🎯 New best model found at round {i}, acc = {current_acc:.2f}")
                    self.save_global_model(round_id=i, best=True)

            for client in self.selected_clients:
                client.train()
                # client.evaluate()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            # if i%self.eval_gap == 0:
            #     print("\nEvaluate global model")
            #     # self.load_model()
            #     self.evaluate()


        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


