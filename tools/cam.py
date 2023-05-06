import argparse
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.optim as optim
from config import cfg
from modeling import build_model

def load_pretrained_model(model,model_path):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print(dir(checkpoint))
    state_dict_ = checkpoint.state_dict()
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}. {}'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    print("checkpoint load finished")
    return model
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        # default='../../reid_data/market1501/bounding_box_test/1398_c5s3_049115_01.jpg',
        default='../cam/26_1.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument(
        "--config_file", default="../configs/AGW_baseline.yml", help="path to config file", type=str
    )
    parser.add_argument('--method', type=str, default='gradcam++',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """
    parser = argparse.ArgumentParser(description="AGW Re-ID Baseline")
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    model = build_model(cfg, 751)

    # pthfile = '../pth/market1501/scp_cat_l1_s/resnet50_scp_nl_model_80.pth'
    pthfile = '../pth/scp_market94.2.pth'
    load_pretrained_model(model, pthfile)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])


    target_layers = [model.base]
    rgb_img = cv2.imread(args.image_path, 1)
    print(rgb_img.shape)

    rgb_img = cv2.resize(rgb_img, (128, 256))
    print(rgb_img.shape)

    # cv2.imshow('img',rgb_img)
    # cv2.waitKey(0)
    rgb_img = rgb_img[:, :, ::-1]
    img = np.ascontiguousarray(rgb_img, dtype=np.float32)
    rgb_img = np.float32(rgb_img) / 255

    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    print(input_tensor.shape)
    result = model(input_tensor)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]
    with cam_algorithm(model=model,
                       target_layers=target_layers,
                       use_cuda=args.use_cuda) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'../cam/{args.method}_cam.jpg', cam_image)
    cv2.imwrite(f'../cam/{args.method}_gb.jpg', gb)
    cv2.imwrite(f'../cam/{args.method}_cam_gb.jpg', cam_gb)

    ####NNI pruner test
    print(model)
    from nni.compression.pytorch.pruning import L1NormPruner
    config_list = [{
        'sparsity_per_layer': 0.5,
        'op_names': ['base.conv1'],
        'op_types': ['Conv2d']
    }, {
        'exclude': False,
        'op_names': ['layer2']
    }]
    pruner = L1NormPruner(model, config_list)
    # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
    # need to unwrap the model, if the model is wrapped before speedup
    pruner._unwrap_model()

    # speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
    from nni.compression.pytorch.speedup import ModelSpeedup

    ModelSpeedup(model, torch.rand(1, 3, 256, 320), masks).speedup_model()
    print(model)
    torch.save(model, './model_l1_pruning.pth')