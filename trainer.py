#%% 
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, date
import time
from itertools import cycle
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, PolynomialLR, StepLR

from losses import *

c2n = True

# Device configuration
with open("parameters/Unet_parameters.json", 'r') as f:
    unet_arguments = json.load(f)
    device = unet_arguments["device"]

with open("parameters/CCT_parameters.json", 'r') as f:
    cct_arguments = json.load(f)

#%% 

class Trainer:
    def __init__(self, model, train_loader, eval_loader, unlabeled_loader=None, arguments=unet_arguments, device=device, timestamp=None):
        
        self.mode = arguments["mode"]
        self.model = model
        self.model.to(device)

        if timestamp is not None: self.timestamp = timestamp
        else: self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logs_name = "logs/logs_unet_" + self.mode + "_" + str(self.timestamp) + ".txt"

        # data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.unlabeled_loader = unlabeled_loader

        self.nb_epochs = arguments["trainer"]["nb_epochs"]

        # supervised loss
        self.sup_loss_mode = arguments["trainer"]["sup_loss_mode"]
        self.supervised_loss = supervised_loss
        self.eval_loss = eval_loss
        
        
        # unsupervised loss and stuff 
        if self.mode == "semi":
            self.unsup_loss_mode = arguments["trainer"]["unsup_loss_mode"]
            self.unsupervised_loss = unsupervised_loss
            self.iter_per_epoch = len(unlabeled_loader)    # assuming that len(unlabeled) > len(labeled)
            self.rampup_length = arguments["trainer"]["rampup_length"]
            self.weight_ul_max = arguments["trainer"]["weight_ul_max"]
        else:
            self.iter_per_epoch = len(train_loader)

        # optimizer 
        if arguments["trainer"]["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=arguments["trainer"]["optimizer_args"]["lr"], momentum=arguments["trainer"]["optimizer_args"]["momentum"])
        elif arguments["trainer"]["optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=arguments["trainer"]["optimizer_args"]["lr"])
        else:
            raise ValueError("optimizer has an invalid value. Must be in ['sgd', 'adam']")
        
        # scheduler
        if arguments["trainer"]["scheduler"] == "OneCycleLR":
            self.scheduler = OneCycleLR(self.optimizer, max_lr = 1e-1, steps_per_epoch = self.iter_per_epoch, epochs = self.nb_epochs, anneal_strategy = 'cos')
        elif arguments["trainer"]["scheduler"] == "PolynomialLR":
            self.scheduler = PolynomialLR(self.optimizer, total_iters=self.nb_epochs, power=0.9)
        elif arguments["trainer"]["scheduler"] == "CosineAnnealingLR":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max = 40, eta_min = 1e-5)
        elif arguments["trainer"]["scheduler"] == "CosineAnnealingWarmRestarts":
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, t_0=5, T_max = 40, eta_min = 1e-6)
        elif arguments["trainer"]["scheduler"] == "StepLR":
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.98)
        else:
            self.scheduler = None
            # raise ValueError("scheduler has an invalid value. Must be in ['OneCycleLR', 'PolynomialLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts]")
        
        # step values
        self.loss_printer_step = arguments["trainer"]["loss_printer_step"]
        self.model_saver_step = arguments["trainer"]["model_saver_step"]



    def train_super_1epoch(self, epoch_idx, tb_writer):
        assert self.mode == "super"
        if not(self.model.training): self.model.train()

        dataloader = iter(self.train_loader)
        
        running_loss = 0.
        last_loss = 0.

        for i, (x, target) in enumerate(dataloader):
            start_time = time.time()

            x = x.to(device)
            target = target.to(device)

            self.optimizer.zero_grad()

            output = self.model(x)

            loss = self.supervised_loss(output, target, mode=self.sup_loss_mode)
            loss.backward()
            self.optimizer.step()            

            # report data
            running_loss += loss.item()
            if i % self.loss_printer_step == 0:
                if i==0: last_loss = running_loss
                else: last_loss = running_loss / self.loss_printer_step
                
                # logs file 
                with open(self.logs_name,"a") as logs :
                    logs.write("\nEpoch : " + str(epoch_idx) + " - batch nb : "+str(i)+" -  in "+ str(int(1000*(time.time()-start_time))) + "ms, loss "+ str(last_loss))
                    logs.close()
                # tensorboard
                print('  batch {} loss: {}'.format(i, last_loss))
                tb_x = epoch_idx * len(self.train_loader) + i
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    

    def train_semi_1epoch(self, epoch_idx, tb_writer):
        assert self.mode == "semi"
        if not(self.model.training): self.model.train()

        dataloader = iter(zip(cycle(self.train_loader), self.unlabeled_loader))

        running_loss = 0.
        running_loss_l = 0.
        running_loss_ul = 0.
        last_loss = 0.

        for i, ((x_l, target_l), x_ul) in enumerate(dataloader):
            start_time = time.time()

            x_l = x_l.to(device)
            target_l = target_l.to(device)
            x_ul = x_ul.to(device)

            self.optimizer.zero_grad()

            logits_l, main_logits_ul, aux_logits_ul = self.model(x_l, x_ul, False)

            # supervised loss
            loss_l = self.supervised_loss(logits_l, target_l, mode=self.sup_loss_mode)

            # unsupervised loss 
            target_ul = F.softmax(main_logits_ul.detach(), dim=1)
            loss_ul = sum([ self.unsupervised_loss(logits, target_ul, mode=self.unsup_loss_mode) for logits in aux_logits_ul ]) / len(aux_logits_ul)
            w_u = weight_ramp_up(self.iter_per_epoch * epoch_idx + i, self.rampup_length, self.weight_ul_max)
            
            loss = loss_l + loss_ul * w_u
            loss.backward()
            self.optimizer.step()

            # report data
            running_loss += loss.item()
            running_loss_l += loss_l.item()
            running_loss_ul += loss_ul.item()
            if i % self.loss_printer_step == 0:
                if i==0: last_loss = running_loss
                else: 
                    last_loss = running_loss / self.loss_printer_step
                    running_loss_l /= 250
                    running_loss_ul /= 250
                # logs file 
                with open(self.logs_name,"a") as logs :
                    logs.write("\nEpoch : " + str(epoch_idx) + " - batch nb : "+str(i)+" -  in "+ str(int(1000*(time.time()-start_time))) + "ms, loss "+ str(last_loss) + " - loss_l " + str(running_loss_l) + " - loss_ul " + str(running_loss_ul) )
                    logs.close()
                # tensorboard
                print('  batch {} loss: {} - loss_l: {} - loss_ul {}'.format(i, last_loss, running_loss_l, running_loss_ul))
                tb_x = epoch_idx * len(self.unlabeled_loader) + i
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
                running_loss_l = 0.
                running_loss_ul = 0.

        return last_loss
    
    
    def eval_1epoch(self, epoch_idx):
        start_time = time.time()
        if self.model.training: self.model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for i, (val_x, val_target) in enumerate(self.eval_loader):
                val_x = val_x.to(device)
                val_target = val_target.to(device)

                val_output = self.model(val_x, eval=True) 
                val_loss = self.eval_loss(val_output, val_target, mode=self.sup_loss_mode)
                running_val_loss += val_loss
        val_loss = running_val_loss / (i+1)

        # report data 
        with open(self.logs_name,"a") as logs :
            logs.write("\nEpoch : " + str(epoch_idx) + " - Eval - in "+ str(int(1000*(time.time()-start_time))) + "ms, val_loss "+ str(val_loss.item()))
            logs.close()

        return  val_loss
    
    def train(self):
        
        writer = SummaryWriter('runs/trainer_unet_{}_{}'.format(self.timestamp, self.mode))
        with open(self.logs_name,"a") as logs :
            logs.write("\n \n")
            logs.write("\nTraining - " + str(self.timestamp) +  "\n")
            logs.close()
        best_val_loss = 1e20

        for epoch_idx in range(self.nb_epochs):
            print('EPOCH {}:'.format(epoch_idx))

            # train on one epoch 
            self.model.train(True)
            if self.mode == "super":  avg_train_loss = self.train_super_1epoch(epoch_idx, writer)
            if self.mode == "semi":  avg_train_loss = self.train_semi_1epoch(epoch_idx, writer)

            # eval after the epoch
            self.model.eval()
            avg_val_loss = self.eval_1epoch(epoch_idx)

            # scheduler 
            if self.scheduler is not None:
                self.scheduler.step()

            # report data
            print('LOSS train {} eval {}'.format(avg_train_loss, avg_val_loss))
            writer.add_scalars('Training vs. Validation Loss', { 'Training' : avg_train_loss, 'Eval' : avg_val_loss }, epoch_idx)
            writer.flush()

            # save (best) models
            if c2n:
                model_path = 'C:/Users/lucas.degeorge/Documents/trained_models/unet_{}_epoch{}.pth'.format(self.timestamp, epoch_idx)
            else:
                model_path = 'trained_models/unet_{}_epoch{}.pth'.format(self.timestamp, epoch_idx)
            if epoch_idx % self.model_saver_step == 0:
                torch.save(self.model.state_dict(), model_path)
            if avg_val_loss < best_val_loss:
                if c2n: model_path = 'C:/Users/lucas.degeorge/Documents/trained_models/unet_{}_best.pth'.format(self.timestamp)
                else: model_path = 'trained_models/unet_{}_best.pth'.format(self.timestamp)
                torch.save(self.model.state_dict(), model_path)
                print("new best epoch: ", epoch_idx)
                best_val_loss = avg_val_loss
