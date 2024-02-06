from torch.optim.lr_scheduler import StepLR, PolynomialLR, CosineAnnealingLR, MultiStepLR, ExponentialLR


def set_scheduler(args, model, optimizer):
    if(args.lr_scheduler == 'poly'):
        total_iters = args.epochs - 5
        lr_scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=args.power)
    elif(args.lr_scheduler == 'step'):
        lr_scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    elif(args.lr_scheduler == 'cos'):
        lr_scheduler = CosineAnnealingLR(optimizer, T_max = args.epochs , eta_min=args.min_lr, verbose=False)
    elif(args.lr_scheduler == 'multistep'):
        lr_scheduler = MultiStepLR(optimizer, milestones=[10,(args.epochs-10)], gamma=args.gamma)
    elif(args.lr_scheduler == 'exp'):
        lr_scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    return lr_scheduler
    
