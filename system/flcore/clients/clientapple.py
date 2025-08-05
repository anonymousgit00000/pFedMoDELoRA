# import copy
# import math
# import torch
# import numpy as np
# import time
# from flcore.clients.clientbase import Client


# class clientAPPLE(Client):
#     def __init__(self, args, id, train_samples, test_samples, **kwargs):
#         super().__init__(args, id, train_samples, test_samples, **kwargs)

#         self.drlr = args.dr_learning_rate
#         self.num_clients = args.num_clients
#         self.lamda = 1
#         self.mu = args.mu
#         self.L = int(args.L * args.global_rounds)
#         self.learning_rate = self.learning_rate * self.num_clients

#         self.model_cs = []

#         self.ps = [1/args.num_clients for _ in range(self.num_clients)]
#         self.p0 = None
#         self.model_c = copy.deepcopy(self.model)

#     def train(self, R):
#         trainloader = self.load_train_data()
        
#         start_time = time.time()

#         self.model.to(self.device)

#         for name, param in self.model.named_parameters():
#             param.requires_grad = "lora_" in name

#         for name, param in self.model_cs[self.id].named_parameters():
#             param.requires_grad = "lora_" in name

#         lora_params = filter(lambda p: p.requires_grad, self.model.parameters())
#         lora_params = list(lora_params)
#         self.optimizer = torch.optim.Adam(lora_params, lr=self.learning_rate)

#         self.model.train()

#         max_local_epochs = self.local_epochs
#         if self.train_slow:
#             max_local_epochs = np.random.randint(1, max_local_epochs // 2)

#         for epoch in range(max_local_epochs):
#             for i, (x, y) in enumerate(trainloader):
#                 if type(x) == type([]):
#                     x[0] = x[0].to(self.device)
#                 else:
#                     x = x.to(self.device)
#                 y = y.to(self.device)
#                 if self.train_slow:
#                     time.sleep(0.1 * np.abs(np.random.rand()))

#                 self.aggregate_parameters()

#                 output = self.model(x)
#                 loss = self.loss(output, y)
#                 self.optimizer.zero_grad()
#                 loss.backward()

#                 for param_c, param in zip(self.model_cs[self.id].parameters(), self.model.parameters()):
#                     param_c.data = param_c - self.learning_rate * param.grad * self.ps[self.id]

#                 for cid in range(self.num_clients):
#                     cnt = 0
#                     p_grad = 0
#                     for param_c, param in zip(self.model_cs[cid].parameters(), self.model.parameters()):
#                         p_grad += torch.mean(param.grad * param_c).item()
#                         cnt += 1
#                     p_grad = p_grad / cnt
#                     p_grad = p_grad + self.lamda * self.mu * (self.ps[cid] - self.p0[cid])
#                     self.ps[cid] = self.ps[cid] - self.drlr * p_grad

#         if R < self.L:
#             self.lamda = (math.cos(R * math.pi / self.L) + 1) / 2
#         else:
#             self.lamda = 0

#         # recover self.model_cs[self.id] for other clients
#         for param_c, param_ in zip(self.model_cs[self.id].parameters(), self.model_c.parameters()):
#             param_c.data = param_.data.clone()

#         self.model_c = copy.deepcopy(self.model)

#         if self.learning_rate_decay:
#             self.learning_rate_scheduler.step()

#         self.train_time_cost['num_rounds'] += 1
#         self.train_time_cost['total_cost'] += time.time() - start_time

        
#     def set_models(self, model_cs):
#         self.model_cs = model_cs

#     def add_parameters(self, w, client_model):
#         for server_param, client_param in zip(self.model.parameters(), client_model.parameters()):
#             server_param.data += client_param.data.clone() * w

#     def aggregate_parameters(self):
#         assert (len(self.model_cs) > 0)

#         for param in self.model.parameters():
#             param.data = torch.zeros_like(param.data)
            
#         for w, client_model in zip(self.ps, self.model_cs):
#             self.add_parameters(w, client_model)


import copy
import math
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAPPLE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.drlr = args.dr_learning_rate
        self.num_clients = args.num_clients
        self.lamda = 1
        self.mu = args.mu
        self.L = int(args.L * args.global_rounds)
        self.learning_rate = self.learning_rate * self.num_clients

        self.model_cs = []

        self.ps = [1/args.num_clients for _ in range(self.num_clients)]
        self.p0 = None
        self.model_c = copy.deepcopy(self.model)

    def train(self, R):
        trainloader = self.load_train_data()

        start_time = time.time()
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # 拉取聚合模型参数
                self.aggregate_parameters()

                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()

                # ✅ 仅打印哪些模块梯度为 None
                none_grad_modules = set()
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is None:
                        module_name = name.split('.')[0]
                        none_grad_modules.add(module_name)
                if none_grad_modules:
                    print(f"[⚠️ client {self.id}] 模块梯度为 None（未参与反向传播或未训练）:")
                    print(f" - {', '.join(sorted(none_grad_modules))}")

                # ============ 更新个性化模型参数 ============
                for param_c, param in zip(self.model_cs[self.id].parameters(), self.model.parameters()):
                    if param.grad is not None:
                        param_c.data = param_c - self.learning_rate * param.grad * self.ps[self.id]

                # ============ 更新权重 ps ============
                for cid in range(self.num_clients):
                    p_grad = 0
                    cnt = 0
                    for param_c, param in zip(self.model_cs[cid].parameters(), self.model.parameters()):
                        if param.grad is not None:
                            p_grad += torch.mean(param.grad * param_c).item()
                            cnt += 1
                    if cnt > 0:
                        p_grad = p_grad / cnt
                        p_grad += self.lamda * self.mu * (self.ps[cid] - self.p0[cid])
                        self.ps[cid] -= self.drlr * p_grad

        if R < self.L:
            self.lamda = (math.cos(R * math.pi / self.L) + 1) / 2
        else:
            self.lamda = 0

        # 恢复当前 client 的个性化模型
        for param_c, param_ in zip(self.model_cs[self.id].parameters(), self.model_c.parameters()):
            param_c.data = param_.data.clone()
        self.model_c = copy.deepcopy(self.model)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_models(self, model_cs):
        self.model_cs = model_cs

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.model_cs) > 0)

        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)

        for w, client_model in zip(self.ps, self.model_cs):
            self.add_parameters(w, client_model)
