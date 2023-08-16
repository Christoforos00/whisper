from datetime import datetime
from transformers import pipeline
from memory_profiler import profile


def predict(processor, model, y, sr):
    # load dummy dataset and read audio files
    input_features = processor(y, sampling_rate=sr, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, max_length=448)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription

# @profile()
def main():
    # Configure base model and save directory for compressed model
    model_id = "openai/whisper-large-v2"
    save_dir = "whisper-large-v2"

    # Number of inferences for comparing timings
    num_inferences = 1
    save_dir = "whisper-large-v2"
    inference_file = "audio30.mp3"

    # Create pipeline with original model as baseline
    cls_pipeline_original = pipeline("automatic-speech-recognition", model=model_id)

    # Measure inference of original model
    start_original = datetime.now()
    if num_inferences == 1:
        res_orig = cls_pipeline_original(inference_file)
    else:
        for i in range(num_inferences):
            res_orig = cls_pipeline_original(inference_file)
    end_original = datetime.now()

    original_inference_time = (end_original - start_original).total_seconds() / num_inferences
    print(f"Original inference time: {original_inference_time}")

    print(res_orig)

    return res_orig
    # original or onnx only 19, quantized 13, 2 iters


if __name__ == "__main__":
    res_orig = main()
