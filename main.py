from trainer import *
from utils import *
from sampler import *
import json

import argparse


def get_params():
    args = argparse.ArgumentParser()
    args.add_argument("-data", "--dataset", default="electronics", type=str)
    args.add_argument("-seed", "--seed", default=None, type=int)
    args.add_argument("-K", "--K", default=3, type=int) #NUMBER OF SHOT

    args.add_argument("-dim", "--embed_dim", default=100, type=int)
    args.add_argument("-bs", "--batch_size", default=1024, type=int)
    args.add_argument("-lr", "--learning_rate", default=0.001, type=float)

    args.add_argument("-epo", "--epoch", default=100000, type=int)
    args.add_argument("-prt_epo", "--print_epoch", default=100, type=int)
    args.add_argument("-eval_epo", "--eval_epoch", default=1000, type=int)

    args.add_argument("-b", "--beta", default=5, type=float)
    args.add_argument("-m", "--margin", default=1, type=float)
    args.add_argument("-p", "--dropout_p", default=0.5, type=float)

    args.add_argument("-gpu", "--device", default=0, type=int)


    args = args.parse_args()
    params = {}
    for k, v in vars(args).items():
        params[k] = v

    params['device'] = torch.device('cuda:'+str(args.device))

    return params, args

if __name__ == '__main__':
    params, args = get_params()

    if params['seed'] is not None:
        SEED = params['seed']
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        np.random.seed(SEED)
        random.seed(SEED)

    user_train, usernum_train, itemnum, user_input_test, user_test, user_input_valid, user_valid = data_load(args.dataset, args.K)    

    sampler = WarpSampler(user_train, usernum_train, itemnum, batch_size=args.batch_size, maxlen=args.K, n_workers=3)

    sampler_test = DataLoader(user_input_test, user_test, itemnum, params)

    sampler_valid = DataLoader(user_input_valid, user_valid, itemnum, params)
    

    trainer = Trainer([sampler, sampler_valid, sampler_test], itemnum, params)

    trainer.train()

    sampler.close()