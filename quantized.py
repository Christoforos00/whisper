from pathlib import Path
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForSpeechSeq2Seq,
    ORTQuantizer
)

# Configure base model and save directory for compressed model
model_id = "openai/whisper-large-v2"
save_dir = "whisper-large"

# Export model in ONNX
model_onnx = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
model_dir = model_onnx.model_save_dir

# Run quantization for all ONNX files of exported model
onnx_models = list(Path(model_dir).glob("*.onnx"))
print(onnx_models)
quantizers = [ORTQuantizer.from_pretrained(model_dir, file_name=onnx_model) for onnx_model in onnx_models]

qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

for quantizer in quantizers:
    # Apply dynamic quantization and save the resulting model
    quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)


from datetime import datetime
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor

# Number of inferences for comparing timings
num_inferences = 4
save_dir = "whisper-large"
inference_file = "audio30.mp3"

# Create pipeline based on simple ONNX model
model_onnx = ORTModelForSpeechSeq2Seq.from_pretrained(model_dir)
tokenizer_onnx = AutoTokenizer.from_pretrained(model_dir)
feature_extractor_onnx = AutoFeatureExtractor.from_pretrained(model_dir)
cls_pipeline_onnx = pipeline("automatic-speech-recognition", model=model_onnx, tokenizer=tokenizer_onnx, feature_extractor=feature_extractor_onnx)

# Create pipeline based on quantized ONNX model
model_quant = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)
tokenizer_quant = AutoTokenizer.from_pretrained(save_dir)
feature_extractor_quant = AutoFeatureExtractor.from_pretrained(save_dir)
cls_pipeline_quant = pipeline("automatic-speech-recognition", model=model_quant, tokenizer=tokenizer_quant, feature_extractor=feature_extractor_quant)

# Create pipeline with original model as baseline
cls_pipeline_original = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")

# Measure inference of quantized model
start_quantized = datetime.now()
for i in range(num_inferences):
    res_quant = cls_pipeline_quant(inference_file)
end_quantized = datetime.now()

# Measure inference of simple onnx model
start_onnx = datetime.now()
for i in range(num_inferences):
    res_onnx = cls_pipeline_onnx(inference_file)
end_onnx = datetime.now()

# Measure inference of original model
start_original = datetime.now()
for i in range(num_inferences):
    res_orig = cls_pipeline_original(inference_file)
end_original = datetime.now()

original_inference_time = (end_original - start_original).total_seconds() / num_inferences
print(f"Original inference time: {original_inference_time}")

quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
print(f"Quantized inference time: {quantized_inference_time}")

onnx_inference_time = (end_onnx - start_onnx).total_seconds() / num_inferences
print(f"Onnx inference time: {onnx_inference_time}")

print(res_onnx)
print(res_quant)
print(res_orig)
print(res_quant == res_orig)
print(res_onnx == res_orig)

# original or onnx only 19, quantized 13, 2 iters