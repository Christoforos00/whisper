
# Configure base model and save directory for compressed model
model_id = "openai/whisper-large-v2"
save_dir = "whisper-large-v2"

from datetime import datetime
from transformers import pipeline

# Number of inferences for comparing timings
num_inferences = 4
save_dir = "whisper-large-v2"
inference_file = "audio30.mp3"

# Create pipeline with original model as baseline
cls_pipeline_original = pipeline("automatic-speech-recognition", model=model_id)

# Measure inference of original model
start_original = datetime.now()
for i in range(num_inferences):
    res_orig = cls_pipeline_original(inference_file)
end_original = datetime.now()

original_inference_time = (end_original - start_original).total_seconds() / num_inferences
print(f"Original inference time: {original_inference_time}")

print(res_orig)


# original or onnx only 19, quantized 13, 2 iters