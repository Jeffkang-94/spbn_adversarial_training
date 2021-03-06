import torch
import torchvision.transforms as T
import torchvision.datasets as datasets
import models
import os.path as osp
import torch.backends.cudnn as cudnn
from attacks import PGDAttacker
from tqdm import tqdm
from torchvision.utils import save_image
import foolbox
class Evaluator:
    def __init__(self, args):

        self.args = args

        transformer = T.Compose([
            T.ToTensor()
        ])
        kwargs = {'num_workers': 4, 'pin_memory': True}

        self.val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.data_root, train=False, transform=transformer),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # Create model, optimizer and scheduler
        self.model = models.WRN(depth=34, width=1, num_classes=10)
        if args.spbn:
            print("SPBN training!")
            self.model = models.convert_splitbn_model(self.model).cuda()
        else:
            self.model.cuda()

        # Loading model
        assert self.args.restore is not None

        model_data = torch.load(self.args.restore)
        self.model.load_state_dict(model_data['model'])
        self.model.eval()

        cudnn.benchmark = True
        self.save_path = self.args.save_path
        self.epsilons = args.epsilon/255.0


        # Foolbox Attack #
        self.model = foolbox.PyTorchModel(self.model, bounds=(0,1))
        if args.attack=='FGSM':
            self.attack = foolbox.attacks.LinfFastGradientAttack(random_start=True)
        elif args.attack=='PGD':
            self.attack = foolbox.attacks.LinfPGD(steps=args.attack_steps, abs_stepsize=2.0/255.0, random_start=True)
        elif args.attack=='BA':
            self.attack = foolbox.attacks.BoundaryAttack()
        elif args.attack=='CW':
            self.attack = foolbox.attacks.L2CarliniWagnerAttack(steps=1000, confidence=20) # confidence == kappa

    def _log(self, message):
        print(message)
        f = open(osp.join(self.save_path, 'log.txt'), 'a+')
        f.write(message + '\n')
        f.close()

    def eval(self):
        if self.args.attack_eps == 0 or self.args.attack_steps == 0:
            acc = self.eval_worker(adv_flag=False)
        else:
            acc = self.eval_worker(adv_flag=True)
        message = 'PGD-{}; Acc:{}'.format(self.args.attack_steps, acc)
        self._log(message)
        return acc

    def eval_worker(self, adv_flag=True):
        clean_correct = 0
        adv_correct=0
        total = 0

        tq = tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=True)
        print(self.args.attack_steps)
        for i, data in tq:
            input, target = data
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            adv_input, _,_ = self.attack(self.model, input, target, epsilons=self.epsilons)

            # compute output
            with torch.no_grad():
                clean_output = self.model(input)
                adv_output = self.model(adv_input)


            _, pred = torch.max(clean_output, dim=1)
            clean_correct += (pred == target).sum()

            _, pred = torch.max(adv_output, dim=1)
            adv_correct += (pred == target).sum()

            total += target.size(0)
            tq.set_description("clean_acc : {:.4f}, adv_acc: {:.4f} ".format(float(clean_correct)/total*100, float(adv_correct)/total*100))

            if i==0:
                save_image(input[:10], "test_image.jpg")

        accuracy = (float(clean_correct) / total) * 100
        return accuracy
