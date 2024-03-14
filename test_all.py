import io
import unittest
import numpy as np

import torch
from torch import nn, Tensor
from typing import List

from models.matcher import HungarianMatcher
from models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from models.backbone import Backbone, Joiner, BackboneBase
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from hubconf import detr_resnet50, detr_resnet50_panoptic

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None


class Tester(unittest.TestCase):

    def test_box_cxcywh_to_xyxy(self):
        t = torch.rand(10, 4)
        r = box_ops.box_xyxy_to_cxcywh(box_ops.box_cxcywh_to_xyxy(t))
        self.assertLess((t - r).abs().max(), 1e-5)

    @staticmethod
    def indices_torch2python(indices):
        return [(i.tolist(), j.tolist()) for i, j in indices]

    def test_hungarian(self):
        n_queries, n_targets, n_classes = 100, 15, 91
        logits = torch.rand(1, n_queries, n_classes + 1)
        boxes = torch.rand(1, n_queries, 4)
        tgt_labels = torch.randint(high=n_classes, size=(n_targets,))
        tgt_boxes = torch.rand(n_targets, 4)
        matcher = HungarianMatcher()
        targets = [{'labels': tgt_labels, 'boxes': tgt_boxes}]
        indices_single = matcher({'pred_logits': logits, 'pred_boxes': boxes}, targets)
        indices_batched = matcher({'pred_logits': logits.repeat(2, 1, 1),
                                   'pred_boxes': boxes.repeat(2, 1, 1)}, targets * 2)
        self.assertEqual(len(indices_single[0][0]), n_targets)
        self.assertEqual(len(indices_single[0][1]), n_targets)
        self.assertEqual(self.indices_torch2python(indices_single),
                         self.indices_torch2python([indices_batched[0]]))
        self.assertEqual(self.indices_torch2python(indices_single),
                         self.indices_torch2python([indices_batched[1]]))

        # test with empty targets
        tgt_labels_empty = torch.randint(high=n_classes, size=(0,))
        tgt_boxes_empty = torch.rand(0, 4)
        targets_empty = [{'labels': tgt_labels_empty, 'boxes': tgt_boxes_empty}]
        indices = matcher({'pred_logits': logits.repeat(2, 1, 1),
                           'pred_boxes': boxes.repeat(2, 1, 1)}, targets + targets_empty)
        self.assertEqual(len(indices[1][0]), 0)
        indices = matcher({'pred_logits': logits.repeat(2, 1, 1),
                           'pred_boxes': boxes.repeat(2, 1, 1)}, targets_empty * 2)
        self.assertEqual(len(indices[0][0]), 0)

    def test_position_encoding_script(self):
        m1, m2 = PositionEmbeddingSine(), PositionEmbeddingLearned()
        mm1, mm2 = torch.jit.script(m1), torch.jit.script(m2)  # noqa

    def test_backbone_script(self):
        backbone = Backbone('resnet50', True, False, False)
        torch.jit.script(backbone)  # noqa

    def test_model_script_detection(self):
        model = detr_resnet50(pretrained=False).eval()
        scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))

    def test_model_script_panoptic(self):
        model = detr_resnet50_panoptic(pretrained=False).eval()
        scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
        out = model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))
        self.assertTrue(out["pred_masks"].equal(out_script["pred_masks"]))

    def test_model_detection_different_inputs(self):
        model = detr_resnet50(pretrained=False).eval()
        # support NestedTensor
        x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
        out = model(x)
        self.assertIn('pred_logits', out)
        # and 4d Tensor
        x = torch.rand(1, 3, 200, 200)
        out = model(x)
        self.assertIn('pred_logits', out)
        # and List[Tensor[C, H, W]]
        x = torch.rand(3, 200, 200)
        out = model([x])
        self.assertIn('pred_logits', out)

    def test_warpped_model_script_detection(self):
        class WrappedDETR(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, inputs: List[Tensor]):
                sample = nested_tensor_from_tensor_list(inputs)
                return self.model(sample)

        model = detr_resnet50(pretrained=False)
        wrapped_model = WrappedDETR(model)
        wrapped_model.eval()
        scripted_model = torch.jit.script(wrapped_model)
        x = [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
        out = wrapped_model(x)
        out_script = scripted_model(x)
        self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))


@unittest.skipIf(onnxruntime is None, 'ONNX Runtime unavailable')
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def compute_boxAP(self, pred_boxes, pred_logits_sigmoid, pred_logits_argmax, target_boxes, target_labels, iou_threshold=0.5):
        """
        Compute the box Average Precision (AP) given the predicted boxes and logits, and the target boxes and labels.
        
        Args:
        - pred_boxes: Predicted bounding boxes (Tensor), shape (N, 4), where N is the number of predicted boxes.
        - pred_logits_sigmoid: Sigmoid of predicted logits (Tensor), shape (N, num_classes), where num_classes is the number of classes.
        - pred_logits_argmax: Argmax of predicted logits (Tensor), shape (N,), representing the predicted class for each box.
        - target_boxes: Target bounding boxes (Tensor), shape (M, 4), where M is the number of target boxes.
        - target_labels: Target labels (Tensor), shape (M,), representing the class for each target box.
        - iou_threshold: IoU threshold for matching predicted and target boxes.

        Returns:
        - boxAP: Box Average Precision (float)
        """

        # Convert tensors to numpy arrays
        pred_boxes_np = pred_boxes.cpu().numpy()
        pred_logits_sigmoid_np = pred_logits_sigmoid.cpu().numpy()
        pred_logits_argmax_np = pred_logits_argmax.cpu().numpy()
        target_boxes_np = target_boxes.cpu().numpy()
        target_labels_np = target_labels.cpu().numpy()

        # Calculate IoU matrix between predicted boxes and target boxes
        iou_matrix = np.zeros((len(pred_boxes_np), len(target_boxes_np)))
        for i in range(len(pred_boxes_np)):
            for j in range(len(target_boxes_np)):
                # Extract individual coordinates
                box1 = pred_boxes_np[i]
                box2 = target_boxes_np[j]
                iou_matrix[i, j] = self.calculate_iou(box1, box2)

        # Initialize variables to store true positives, false positives, and false negatives
        tp = 0
        fp = 0
        fn = 0

        # Match predicted boxes to target boxes using the Hungarian algorithm
        matched_indices = self.hungarian_matching(iou_matrix)

        # Loop through each matched pair of indices
        for pred_idx, target_idx in matched_indices:
            # Check if the IoU is above the threshold and the predicted class matches the target class
            if iou_matrix[pred_idx, target_idx] >= iou_threshold and pred_logits_argmax_np[pred_idx] == target_labels_np[target_idx]:
                tp += 1  # True positive
            else:
                fp += 1  # False positive

        # Calculate false negatives by subtracting true positives from the total number of target boxes
        fn = len(target_boxes_np) - tp

        # Calculate precision and recall
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        # Calculate Average Precision (AP)
        # For simplicity, let's assume AP is just precision at this stage
        boxAP = precision

        return boxAP

    def calculate_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Args:
        - box1: Bounding box coordinates (xmin, ymin, xmax, ymax)
        - box2: Bounding box coordinates (xmin, ymin, xmax, ymax)

        Returns:
        - iou: Intersection over Union (IoU) between the two boxes
        """
        # Calculate intersection coordinates
        xmin_inter = max(box1[0], box2[0])
        ymin_inter = max(box1[1], box2[1])
        xmax_inter = min(box1[2], box2[2])
        ymax_inter = min(box1[3], box2[3])

        # Calculate intersection area
        intersection_area = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)

        # Calculate area of each box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Calculate union area
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou

    def hungarian_matching(self, iou_matrix):
        """
        Perform matching between predicted boxes and target boxes using the Hungarian algorithm.

        Args:
        - iou_matrix: Matrix containing IoU scores between predicted boxes and target boxes

        Returns:
        - matched_indices: List of matched indices [(pred_idx, target_idx)]
        """
        # Perform Hungarian matching using scipy.optimize.linear_sum_assignment
        from scipy.optimize import linear_sum_assignment
        pred_indices, target_indices = linear_sum_assignment(-iou_matrix)  # Minimize negative IoU for maximization

        # Create list of matched indices
        matched_indices = [(pred_idx, target_idx) for pred_idx, target_idx in zip(pred_indices, target_indices)]

        return matched_indices

    def run_model(self, model, inputs_list, tolerate_small_mismatch=False, do_constant_folding=True, dynamic_axes=None,
                  output_names=None, input_names=None):
        model.eval()

        onnx_io = io.BytesIO()
        # export to onnx with the first input
        torch.onnx.export(model, inputs_list[0], onnx_io,
                          do_constant_folding=do_constant_folding, opset_version=12,
                          dynamic_axes=dynamic_axes, input_names=input_names, output_names=output_names)
        # validate the exported model with onnx runtime
        outputs = []
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or isinstance(test_inputs, list):
                    test_inputs = (nested_tensor_from_tensor_list(test_inputs),)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)

            print("Test Inputs:", test_inputs)
            print("Expected Outputs:", test_ouputs)
            print("Exported ONNX Model:")
            outputs.append(test_ouputs)
            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)
        
        return outputs

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction 
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        for i, element in enumerate(outputs):
            try:
                torch.testing.assert_allclose(element, ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

            print("Expected Output:", element)
            print("ONNX Output:", ort_outs[i])

    def test_model_onnx_detection(self):
        model = detr_resnet50(pretrained=False).eval()
        dummy_image = torch.ones(1, 3, 800, 800) * 0.3
        model(dummy_image)

        # Test exported model on images of different size, or dummy input
        outputs = self.run_model(
            model,
            [(torch.rand(1, 3, 750, 800),)],
            input_names=["inputs"],
            output_names=["pred_logits", "pred_boxes"],
            tolerate_small_mismatch=True,
        )

        # Extract relevant outputs
        pred_logits = outputs[0]["pred_logits"]
        pred_boxes = outputs[0]["pred_boxes"]

        # Assume you have ground truth boxes and labels available
        target_boxes = torch.rand(5, 4)  
        target_labels = torch.randint(1, 91, (5,))  

        # Calculate boxAP
        boxAP = self.compute_boxAP(pred_boxes, pred_logits.sigmoid(), pred_logits.argmax(-1), target_boxes, target_labels)
        
        print("BoxAP:", boxAP)

        # Assert if required
        self.assertTrue(boxAP > 0.5) 

    @unittest.skip("CI doesn't have enough memory")
    def test_model_onnx_detection_panoptic(self):
        model = detr_resnet50_panoptic(pretrained=False).eval()
        dummy_image = torch.ones(1, 3, 800, 800) * 0.3
        model(dummy_image)

        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(torch.rand(1, 3, 750, 800),)],
            input_names=["inputs"],
            output_names=["pred_logits", "pred_boxes", "pred_masks"],
            tolerate_small_mismatch=True,
        )


if __name__ == '__main__':
    unittest.main()
