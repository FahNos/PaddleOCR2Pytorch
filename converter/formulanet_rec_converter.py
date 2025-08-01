# https://zhuanlan.zhihu.com/p/335753926
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import OrderedDict
import numpy as np
import cv2
import torch
from pytorchocr.base_ocr_v20 import BaseOCRV20

def print_cmp(inp, name=None):
    print('{}: shape-{}, sum: {}, mean: {}, max: {}, min: {}'.format(name, inp.shape,
                                                                     np.sum(inp), np.mean(inp),
                                                                     np.max(inp), np.min(inp)))

class FormulanetRecConverter(BaseOCRV20):
    def __init__(self, config, paddle_pretrained_model_path, **kwargs):
        para_state_dict, opti_state_dict = self.read_paddle_weights(paddle_pretrained_model_path)
        super(FormulanetRecConverter, self).__init__(config)
        self.load_paddle_weights([para_state_dict, opti_state_dict])
        print('model is loaded: {}'.format(paddle_pretrained_model_path))
        self.net.eval()


    def del_invalid_state_dict(self, para_state_dict):
        new_state_dict = OrderedDict()
        for i, (k,v) in enumerate(para_state_dict.items()):
            if k.startswith('head.gtc_head.'):
                continue

            elif k.startswith('head.before_gtc'):
                continue

            else:
                new_state_dict[k] = v
        return new_state_dict


    def load_paddle_weights(self, paddle_weights):
        para_state_dict, opti_state_dict = paddle_weights
   
        for k, v in para_state_dict.items():
            ptname = k
            ptname = ptname.replace('._mean', '.running_mean')
            ptname = ptname.replace('._variance', '.running_var')

            try:
                if any(k.endswith(suffix) for suffix in [
                    'fc1.weight', 'fc2.weight', 'fc.weight', 
                    'qkv.weight', 'proj.weight', 'lm_head.weight',
                    'enc_to_dec_proj.weight'
                ]):
                    self.net.state_dict()[ptname].copy_(torch.from_numpy(v.numpy()).T)
                else:
                    self.net.state_dict()[ptname].copy_(torch.from_numpy(v.numpy()))

            except Exception as e:
                print('exception:')
                print('pytorch: {}, {}'.format(ptname, self.net.state_dict()[ptname].size()))
                print('paddle: {}, {}'.format(k, v.shape))
                raise e

        print('model is loaded.')

def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError('{} is not existed.'.format(yaml_path))
    import yaml
    with open(yaml_path, encoding='utf-8') as f:
        res = yaml.safe_load(f)
    if res.get('Architecture') is None:
        raise ValueError('{} has no Architecture'.format(yaml_path))    
    return res['Architecture']

if __name__ == '__main__':
    import argparse, json, textwrap, sys, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, help='Assign the yaml path of network configuration', default=None)
    parser.add_argument("--src_model_path", type=str, help='Assign the paddleOCR trained model(best_accuracy)')
    args = parser.parse_args()

    yaml_path = args.yaml_path
    if yaml_path is not None:
        if not os.path.exists(yaml_path):
            raise FileNotFoundError('{} is not existed.'.format(yaml_path))
        cfg = read_network_config_from_yaml(yaml_path)

    else:
        raise NotImplementedError

    converter = FormulanetRecConverter(cfg, args.src_model_path)

    np.random.seed(666)
    inputs = np.random.randn(1,1,348,348).astype(np.float32)
    inp = torch.from_numpy(inputs)

    out = converter.net(inp)
    out = out.data.numpy()
    # print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    save_basename = os.path.basename(os.path.abspath(args.src_model_path))
    save_name = 'ptocr_v5_{}.pth'.format(save_basename.split('PP-OCRv5_')[-1].split('_pretrained')[0])
    converter.save_pytorch_weights(save_name)
    print('done.')
