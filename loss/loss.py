#！/usr/bin/env python
#!-*coding:utf-8 -*-
#!@Author :lzy
#!@File :loss.py

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchvision

class CbLoss(nn.Module):
    def __init__(self, epsilon=0.001):
        super(CbLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon ** 2))


class BasicBlock(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), stride=stride, bias=bias)]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class Discriminator(nn.Module):
    def __init__(self, args, gan_type='GAN'):
        super(Discriminator, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        bn = not gan_type == 'WGAN_GP'
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        m_features = [
            BasicBlock(in_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = args.patch_size // (2 ** ((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size ** 2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))
        return output


class Temporal_Discriminator(nn.Module):
    def __init__(self, args):
        super(Temporal_Discriminator, self).__init__()

        in_channels = 3
        out_channels = 64
        depth = 7
        bn = False
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.feature_3d = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1)),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        )

        m_features = [
            BasicBlock(out_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)
        patch_size = args.patch_size // (2 ** ((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size ** 2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, f0, f1, f2):
        f0 = torch.unsqueeze(f0, dim=2)
        f1 = torch.unsqueeze(f1, dim=2)
        f2 = torch.unsqueeze(f2, dim=2)

        x_5d = torch.cat((f0, f1, f2), dim=2)

        x = torch.squeeze(self.feature_3d(x_5d))
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))
        return output


class FI_Discriminator(nn.Module):
    def __init__(self, args):
        super(FI_Discriminator, self).__init__()

        in_channels = 6
        out_channels = 64
        depth = 7
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        m_features = [
            BasicBlock(in_channels, out_channels, 3, bn=bn, act=act)
        ]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(BasicBlock(
                in_channels, out_channels, 3, stride=stride, bn=bn, act=act
            ))

        self.features = nn.Sequential(*m_features)

        patch_size = args.patch_size // (2 ** ((depth + 1) // 2))
        m_classifier = [
            nn.Linear(out_channels * patch_size ** 2, 1024),
            act,
            nn.Linear(1024, 1)
        ]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, f0, f1):
        x = torch.cat((f0, f1), dim=1)
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))
        return output

def Optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': 0.9}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'ADAMax':
        optimizer_function = optim.Adamax
        kwargs = {
            'betas': (0.9, 0.999),
            'eps': 1e-08
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': 1e-08}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)

def Scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(my_optimizer,step_size=args.lr_decay,gamma=args.gamma)
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(my_optimizer,milestones=milestones,gamma=args.gamma)
    return scheduler

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = 1
        if gan_type == 'T_WGAN_GP':
            self.discriminator = Temporal_Discriminator(args)
        elif gan_type == 'FI_GAN':
            self.discriminator = FI_Discriminator(args)
        else:
            self.discriminator = Discriminator(args, gan_type)
        if gan_type != 'WGAN_GP' and gan_type != 'T_WGAN_GP':
            self.optimizer = Optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(self.discriminator.parameters(),betas=(0, 0.9), eps=1e-8, lr=1e-5)
        self.scheduler = Scheduler(args, self.optimizer)

    def forward(self, fake, real, input_frames=None):
        fake_detach = fake.detach()

        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            if self.gan_type == 'T_WGAN_GP':
                d_fake = self.discriminator(input_frames[0], fake_detach, input_frames[1])
                d_real = self.discriminator(input_frames[0], real, input_frames[1])
            elif self.gan_type == 'FI_GAN':
                d_01 = self.discriminator(input_frames[0], fake_detach)
                d_12 = self.discriminator(fake_detach, input_frames[1])
            else:
                d_fake = self.discriminator(fake_detach)
                d_real = self.discriminator(real)

            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type == 'FI_GAN':
                label_01 = torch.zeros_like(d_01)
                label_12 = torch.ones_like(d_12)
                loss_d = F.binary_cross_entropy_with_logits(d_01, label_01) + F.binary_cross_entropy_with_logits(d_12, label_12)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand_like(fake)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            # Discriminator update
            self.loss += loss_d.item()
            loss_d.backward()
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        if self.gan_type == 'GAN':
            d_fake_for_g = self.discriminator(fake)
            loss_g = F.binary_cross_entropy_with_logits(
                d_fake_for_g, label_real
            )
        elif self.gan_type == 'FI_GAN':
            d_01_for_g = F.sigmoid(self.discriminator(input_frames[0], fake_detach))
            d_12_for_g = F.sigmoid(self.discriminator(fake_detach, input_frames[1]))
            loss_g = d_01_for_g * torch.log(d_01_for_g + 1e-12) + d_12_for_g * torch.log(d_12_for_g + 1e-12)
            loss_g = loss_g.mean()

        elif self.gan_type.find('WGAN') >= 0:
            d_fake_for_g = self.discriminator(fake)
            loss_g = -d_fake_for_g.mean()

        # Generator loss
        return loss_g

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16_conv_4_3 = torch.nn.Sequential(*list(vgg16.children())[0][:22])
        for param in self.vgg16_conv_4_3.parameters():
            param.requires_grad = False

    def forward(self, output, gt):
        vgg_output = self.vgg16_conv_4_3(output)
        with torch.no_grad():
            vgg_gt = self.vgg16_conv_4_3(gt.detach())
        loss = F.mse_loss(vgg_output, vgg_gt)
        return loss

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.loss = []
        self.loss_module = nn.ModuleList()
        self.regularize = []

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'Charb':
                loss_function = CbLoss()
            elif loss_type.find('VGG') >= 0:
                loss_function =  VGG()
            elif loss_type.find('GAN') >= 0:
                loss_function = Adversarial(args,loss_type)
            elif loss_type in ['g_Spatial', 'g_Occlusion', 'Lw', 'Ls']:
                self.regularize.append({'type': loss_type,'weight': float(weight)})
                continue

            self.loss.append({'type': loss_type,'weight': float(weight),'function': loss_function})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        for r in self.regularize:
            print('{:.3f} * {}'.format(r['weight'], r['type']))

        self.loss_module.to('cuda')

    def forward(self, output, gt, input_frames):
        losses = []
        for l in self.loss:
            if l['function'] is not None:
                if l['type'] == 'T_WGAN_GP' or l['type'] == 'FI_GAN':
                    loss = l['function'](output['frame1'], gt, input_frames)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)
                else:
                    loss = l['function'](output['frame1'], gt)
                    effective_loss = l['weight'] * loss
                    losses.append(effective_loss)
        for r in self.regularize:
            effective_loss = r['weight'] * output[r['type']]
            losses.append(sum(effective_loss))
        loss_sum = sum(losses)
        return loss_sum

