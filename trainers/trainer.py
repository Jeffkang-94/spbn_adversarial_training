import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import models
import torch.optim as optim
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time
import utils
from attacks import PGDAttacker
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 
class Trainer:
    def __init__(self, args):

        self.args = args

        # Creating data loaders
        transform_train = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        transform_test = T.Compose([
            T.ToTensor()
        ])

        kwargs = {'num_workers': 8, 'pin_memory': True}

        self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=True, download=True,
                             transform=transform_train),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, transform=transform_test),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # Create model, optimizer and scheduler
        self.model = models.WRN(depth=34, width=1, num_classes=10)

        
        self.spbn_flag=args.spbn
        if self.spbn_flag:
            print("SPBN training!")
            self.model = models.convert_splitbn_model(self.model, momentum=0.5).cuda()
        else:
            self.model.cuda()

        self.lambda_ = 0.9

        # spbn_1 = 0.7 adv momentum = 0.1
        # spbn_2 = 0.7, adv_momentum = 0.01
        # spbn_3 = 0.9, adv_momentum = 0.01
        # spbn_4 = 0.9, adv_momentum = 0.5
            

        self.optimizer = optim.SGD(self.model.parameters(), args.lr,
                                   momentum=0.9, weight_decay=args.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.1)

        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))


        self.save_path = args.save_path
        self.epoch = 0
        cudnn.benchmark = True
        self.attacker = PGDAttacker(args.attack_eps)

        # resume from checkpoint
        if args.resume:
            ckpt_path = os.path.join(args.save_path, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                self._load_from_checkpoint(ckpt_path)
            elif args.restore:
                self._load_from_checkpoint(args.restore)

        

    def _log(self, message):
        print(message)
        f = open(os.path.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def _load_from_checkpoint(self, ckpt_path):
        print('Loading model from {} ...'.format(ckpt_path))
        model_data = torch.load(ckpt_path)
        self.model.load_state_dict(model_data['model'])
        self.optimizer.load_state_dict(model_data['optimizer'])
        self.lr_scheduler.load_state_dict(model_data['lr_scheduler'])
        self.epoch = model_data['epoch'] + 1
        print('Model loaded successfully')

    def _save_checkpoint(self, best=True):
        self.model.eval()
        model_data = dict()
        model_data['model'] = self.model.state_dict()
        model_data['optimizer'] = self.optimizer.state_dict()
        model_data['lr_scheduler'] = self.lr_scheduler.state_dict()
        model_data['epoch'] = self.epoch
        if best:
            torch.save(model_data, os.path.join(self.save_path, 'best.pth'))
        else:
            torch.save(model_data, os.path.join(self.save_path, 'checkpoint.pth'))

    def train(self):

        losses = utils.AverageMeter()
        # summary writer
        log_dir = self.save_path +'/training_log'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)
        best_acc = 0 
        while self.epoch < self.args.nepochs:
            self.model.train()
            correct = 0
            total = 0
            start_time = time.time()
            tq = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True)
            for i, data in tq:
                input, target = data
                target = target.cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)
                adv_input = self.attacker.attack(input, target, self.model, self.args.attack_steps, self.args.attack_lr,random_init=True, target=None)

                # compute output
                self.optimizer.zero_grad()

                if self.spbn_flag:
                    concat = torch.cat((input, adv_input), dim=0)
                    logits = self.model(concat)
                    clean_logits,adv_logits = torch.split(logits, target.size(0), dim=0)

                    adv_loss = F.cross_entropy(adv_logits, target)
                    clean_loss = F.cross_entropy(clean_logits, target)

                    loss =  self.lambda_* adv_loss + (1-self.lambda_) * clean_loss
                else:
                    clean_logits = self.model(input)
                    loss = F.cross_entropy(clean_logits, target)
                    mean, std = models.print_mean_std(self.model)

                loss.backward()
                self.optimizer.step()

                _, pred = torch.max(clean_logits, dim=1)
                correct += (pred == target).sum()
                total += target.size(0)
                acc = (float(correct) / total) * 100

                # measure accuracy and record loss
                losses.update(loss.data.item(), input.size(0))
                message = 'Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.4f}'.format(self.epoch,self.args.nepochs, loss.item(), acc)
                tq.set_description(message)

                # writing log in Tensorboard
                if self.spbn_flag:
                    self.writer.add_scalars('Adv_training/loss',{'clean_loss':clean_loss.item(),
                                                                'adv_loss': adv_loss.item(),
                                                                'entire_loss': loss.item()})
                    self.writer.add_scalar('Adv_training/Acc',acc)
                else:
                    self.writer.add_scalar('Adv_training/loss', loss.item())
                    self.writer.add_scalar('Adv_training/Acc',acc)

            self.epoch += 1
            self.lr_scheduler.step()
            end_time = time.time()
            batch_time = end_time - start_time
            message = 'Epoch {}, Time {}, Loss: {}, Accuracy: {}'.format(self.epoch, batch_time, loss.item(), acc)
            self._log(message)
            self._save_checkpoint()

            # Evaluation
            if self.epoch%5==0:
                nat_acc = self.eval()
                adv_acc = self.eval_adversarial()

                if adv_acc > best_acc:
                    print("Saving..")
                    self._save_checkpoint(best=True)
                    best_acc = adv_acc
                self._log('Natural accuracy: {}'.format(nat_acc))
                self._log('Adv accuracy: {}'.format(adv_acc))

            self._save_checkpoint(best=False)

            

    def eval(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            input, target = data
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            # compute output
            with torch.no_grad():
                output = self.model(input)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)

        accuracy = (float(correct) / total) * 100
        return accuracy

    def eval_adversarial(self):
        self.model.eval()

        correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            input, target = data
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            input = self.attacker.attack(input, target, self.model, self.args.attack_steps, self.args.attack_lr,
                                         random_init=True)

            # compute output
            with torch.no_grad():
                output = self.model(input)

            _, pred = torch.max(output, dim=1)
            correct += (pred == target).sum()
            total += target.size(0)

        accuracy = (float(correct) / total) * 100
        return accuracy



