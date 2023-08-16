from pathlib import Path

import onnxruntime
import torch
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSpeechSeq2Seq,
    ORTQuantizer
)
from datetime import datetime
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor

from memory_profiler import profile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from onnxruntime.quantization import quantize_dynamic, QuantType

import librosa

quantize = True
inference = True
num_inferences = 1


@profile()
def main():
    inference_file = "audio30.mp3"

    if not inference:
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
        model_save_dir = "whisper_onnx"
        quant_model_save_dir = "whisper_onnx_quant"
        y, sr = librosa.load(inference_file, sr=16000)
        demo_sample = processor(y, sampling_rate=sr, return_tensors="pt").input_features

        torch.onnx.export(model, demo_sample, model_save_dir, export_params=True, do_constant_folding=True,
                          verbose=False, input_names=['input'], output_names=['output'], opset_version=13,
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})

        quantized_model = quantize_dynamic(model_save_dir, quant_model_save_dir, weight_type=QuantType.QUInt8)
    else:
        y, sr = librosa.load(inference_file, sr=16000)
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")

        inputs = processor(y, sampling_rate=sr, return_tensors="pt").input_features
        model = onnxruntime.InferenceSession("whisper-large-v2")
        ort_inputs = {model["model"].get_inputs()[0].name: inputs.detach().cpu().numpy()}
        ort_outs = model["model"].run(None, ort_inputs)
        encodings_batch = ort_outs[1]


if __name__ == "__main__":
    main()
