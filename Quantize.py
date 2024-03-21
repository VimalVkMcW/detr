import onnxruntime
from onnxruntime import quantization

class OnnxStaticQuantization:
    def __init__(self) -> None:
        self.enum_data = None
        self.calibration_technique = {
            "MinMax": onnxruntime.quantization.calibrate.CalibrationMethod.MinMax,
            "Entropy": onnxruntime.quantization.calibrate.CalibrationMethod.Entropy,
            "Percentile": onnxruntime.quantization.calibrate.CalibrationMethod.Percentile,
            "Distribution": onnxruntime.quantization.calibrate.CalibrationMethod.Distribution
        }
    def get_next(self, EP_list = ['CPUExecutionProvider']):
        if self.enum_data is None:
            session = onnxruntime.InferenceSession(self.fp32_onnx_path, providers=EP_list)
            input_name = session.get_inputs()[0].name
            calib_list = []
            count = 0
            for nhwc_data, _ in self.calibration_loader:
                nhwc_data=nhwc_data.cpu()
                calib_list.append({input_name: nhwc_data.numpy()})
                if self.sample == count: break
                count = count + 1
            self.enum_data = iter(calib_list)
        return next(self.enum_data, None)
    def quantization(self, fp32_onnx_path, future_int8_onnx_path, calib_method, calibration_loader, sample=100):
        self.sample = sample
        self.calibration_loader = calibration_loader
        _ = ort.quantization.quantize_static(
                model_input=fp32_onnx_path,
                model_output=future_int8_onnx_path,
                activation_type=ort.quantization.QuantType.QInt16, weight_type=ort.quantization.QuantType.QInt8,
                calibrate_method=self.calibration_technique[calib_method],
                calibration_data_reader=self
            )
        return self