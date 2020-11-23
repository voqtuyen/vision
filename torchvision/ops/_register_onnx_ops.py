import sys
import torch

_onnx_opset_version = 11


def _register_custom_op():
    from torch.onnx.symbolic_helper import parse_args, scalar_type_to_onnx
    from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze, _cast_Long, reshape

    @parse_args('v', 'v', 'f')
    def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op('Constant', value_t=torch.tensor([sys.maxsize], dtype=torch.long))
        iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores, max_output_per_class, iou_threshold)
        return squeeze(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)

    @parse_args('v', 'v', 'f', 'i', 'i', 'i')
    def roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
        batch_indices = _cast_Long(g, squeeze(g, select(g, rois, 1, g.op('Constant',
                                   value_t=torch.tensor([0], dtype=torch.long))), 1), False)
        rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        return g.op('RoiAlign', input, rois, batch_indices, spatial_scale_f=spatial_scale,
                    output_height_i=pooled_height, output_width_i=pooled_width, sampling_ratio_i=sampling_ratio)

    @parse_args('v', 'v', 'f', 'i', 'i')
    def roi_pool(g, input, rois, spatial_scale, pooled_height, pooled_width):
        roi_pool = g.op('MaxRoiPool', input, rois,
                        pooled_shape_i=(pooled_height, pooled_width), spatial_scale_f=spatial_scale)
        return roi_pool, None

    @parse_args('v', 'v', 'v', 'v', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i')
    def deform_conv2d(g, input, offset, weight, bias, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups):
        return g.op('DeformConv', input, offset, weight, bias, stride_i=(stride_h, stride_w), 
                        padding_i=(pad_h, pad_w), dilation_i=(dilation_h, dilation_w), groups_i=groups, offset_groups_i=offset_groups)

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('torchvision::nms', symbolic_multi_label_nms, _onnx_opset_version)
    register_custom_op_symbolic('torchvision::roi_align', roi_align, _onnx_opset_version)
    register_custom_op_symbolic('torchvision::roi_pool', roi_pool, _onnx_opset_version)
    register_custom_op_symbolic('torchvision::deform_conv2d', deform_conv2d, _onnx_opset_version)
