from common_utils import TestCase, map_nested_tensor_object, freeze_rng_state
from collections import OrderedDict
from itertools import product
import torch
import numpy as np
from torchvision import models
import unittest
import traceback
import random


def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_available_classification_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_segmentation_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.segmentation.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_detection_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.detection.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


def get_available_video_models():
    # TODO add a registration mechanism to torchvision.models
    return [k for k, v in models.video.__dict__.items() if callable(v) and k[0].lower() == k[0] and k[0] != "_"]


# models that are in torch hub, as well as r3d_18. we tried testing all models
# but the test was too slow. not included are detection models, because
# they are not yet supported in JIT.
# If 'unwrapper' is provided it will be called with the script model outputs
# before they are compared to the eager model outputs. This is useful if the
# model outputs are different between TorchScript / Eager mode
script_test_models = {
    'deeplabv3_resnet101': {},
    'mobilenet_v2': {},
    'resnext50_32x4d': {},
    'fcn_resnet101': {},
    'googlenet': {
        'unwrapper': lambda x: x.logits
    },
    'densenet121': {},
    'resnet18': {},
    'alexnet': {},
    'shufflenet_v2_x1_0': {},
    'squeezenet1_0': {},
    'vgg11': {},
    'inception_v3': {
        'unwrapper': lambda x: x.logits
    },
    'r3d_18': {},
    "fasterrcnn_resnet50_fpn": {
        'unwrapper': lambda x: x[1]
    },
    "maskrcnn_resnet50_fpn": {
        'unwrapper': lambda x: x[1]
    },
    "keypointrcnn_resnet50_fpn": {
        'unwrapper': lambda x: x[1]
    },
}


class ModelTester(TestCase):
    def checkModule(self, model, name, args):
        if name not in script_test_models:
            return
        unwrapper = script_test_models[name].get('unwrapper', None)
        return super(ModelTester, self).checkModule(model, args, unwrapper=unwrapper, skip=False)

    def _test_classification_model(self, name, input_shape):
        set_rng_seed(0)
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.__dict__[name](num_classes=50)
        model.eval()
        x = torch.rand(input_shape)
        out = model(x)
        self.assertExpected(out, prec=0.1)
        self.assertEqual(out.shape[-1], 50)
        self.checkModule(model, name, (x,))

    def _test_segmentation_model(self, name):
        # passing num_class equal to a number other than 1000 helps in making the test
        # more enforcing in nature
        model = models.segmentation.__dict__[name](num_classes=50, pretrained_backbone=False)
        model.eval()
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)
        out = model(x)
        self.assertEqual(tuple(out["out"].shape), (1, 50, 300, 300))
        self.checkModule(model, name, (x,))

    def _test_detection_model(self, name):
        set_rng_seed(0)
        model = models.detection.__dict__[name](num_classes=50, pretrained_backbone=False)
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)
        self.assertEqual(len(out), 1)

        def subsample_tensor(tensor):
            num_elems = tensor.numel()
            num_samples = 20
            if num_elems <= num_samples:
                return tensor

            flat_tensor = tensor.flatten()
            ith_index = num_elems // num_samples
            return flat_tensor[ith_index - 1::ith_index]

        def compute_mean_std(tensor):
            # can't compute mean of integral tensor
            tensor = tensor.to(torch.double)
            mean = torch.mean(tensor)
            std = torch.std(tensor)
            return {"mean": mean, "std": std}

        # maskrcnn_resnet_50_fpn numerically unstable across platforms, so for now
        # compare results with mean and std
        if name == "maskrcnn_resnet50_fpn":
            test_value = map_nested_tensor_object(out, tensor_map_fn=compute_mean_std)
            # mean values are small, use large prec
            self.assertExpected(test_value, prec=.01)
        else:
            self.assertExpected(map_nested_tensor_object(out, tensor_map_fn=subsample_tensor), prec=0.01)

        scripted_model = torch.jit.script(model)
        scripted_model.eval()
        scripted_out = scripted_model(model_input)[1]
        self.assertEqual(scripted_out[0]["boxes"], out[0]["boxes"])
        self.assertEqual(scripted_out[0]["scores"], out[0]["scores"])
        # labels currently float in script: need to investigate (though same result)
        self.assertEqual(scripted_out[0]["labels"].to(dtype=torch.long), out[0]["labels"])
        self.assertTrue("boxes" in out[0])
        self.assertTrue("scores" in out[0])
        self.assertTrue("labels" in out[0])
        # don't check script because we are compiling it here:
        # TODO: refactor tests
        # self.check_script(model, name)
        self.checkModule(model, name, ([x],))

    def _test_video_model(self, name):
        # the default input shape is
        # bs * num_channels * clip_len * h *w
        input_shape = (1, 3, 4, 112, 112)
        # test both basicblock and Bottleneck
        model = models.video.__dict__[name](num_classes=50)
        model.eval()
        x = torch.rand(input_shape)
        out = model(x)
        self.checkModule(model, name, (x,))
        self.assertEqual(out.shape[-1], 50)

    def _make_sliced_model(self, model, stop_layer):
        layers = OrderedDict()
        for name, layer in model.named_children():
            layers[name] = layer
            if name == stop_layer:
                break
        new_model = torch.nn.Sequential(layers)
        return new_model

    def test_memory_efficient_densenet(self):
        input_shape = (1, 3, 300, 300)
        x = torch.rand(input_shape)

        for name in ['densenet121', 'densenet169', 'densenet201', 'densenet161']:
            model1 = models.__dict__[name](num_classes=50, memory_efficient=True)
            params = model1.state_dict()
            model1.eval()
            out1 = model1(x)
            out1.sum().backward()

            model2 = models.__dict__[name](num_classes=50, memory_efficient=False)
            model2.load_state_dict(params)
            model2.eval()
            out2 = model2(x)

            max_diff = (out1 - out2).abs().max()

            self.assertTrue(max_diff < 1e-5)

    def test_resnet_dilation(self):
        # TODO improve tests to also check that each layer has the right dimensionality
        for i in product([False, True], [False, True], [False, True]):
            model = models.__dict__["resnet50"](replace_stride_with_dilation=i)
            model = self._make_sliced_model(model, stop_layer="layer4")
            model.eval()
            x = torch.rand(1, 3, 224, 224)
            out = model(x)
            f = 2 ** sum(i)
            self.assertEqual(out.shape, (1, 2048, 7 * f, 7 * f))

    def test_mobilenetv2_residual_setting(self):
        model = models.__dict__["mobilenet_v2"](inverted_residual_setting=[[1, 16, 1, 1], [6, 24, 2, 2]])
        model.eval()
        x = torch.rand(1, 3, 224, 224)
        out = model(x)
        self.assertEqual(out.shape[-1], 1000)

    def test_fasterrcnn_double(self):
        model = models.detection.fasterrcnn_resnet50_fpn(num_classes=50, pretrained_backbone=False)
        model.double()
        model.eval()
        input_shape = (3, 300, 300)
        x = torch.rand(input_shape, dtype=torch.float64)
        model_input = [x]
        out = model(model_input)
        self.assertIs(model_input[0], x)
        self.assertEqual(len(out), 1)
        self.assertTrue("boxes" in out[0])
        self.assertTrue("scores" in out[0])
        self.assertTrue("labels" in out[0])


for model_name in get_available_classification_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        input_shape = (1, 3, 224, 224)
        if model_name in ['inception_v3']:
            input_shape = (1, 3, 299, 299)
        self._test_classification_model(model_name, input_shape)

    setattr(ModelTester, "test_" + model_name, do_test)


for model_name in get_available_segmentation_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        self._test_segmentation_model(model_name)

    setattr(ModelTester, "test_" + model_name, do_test)


for model_name in get_available_detection_models():
    # for-loop bodies don't define scopes, so we have to save the variables
    # we want to close over in some way
    def do_test(self, model_name=model_name):
        self._test_detection_model(model_name)

    setattr(ModelTester, "test_" + model_name, do_test)


for model_name in get_available_video_models():

    def do_test(self, model_name=model_name):
        self._test_video_model(model_name)

    setattr(ModelTester, "test_" + model_name, do_test)

if __name__ == '__main__':
    unittest.main()
