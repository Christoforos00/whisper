from pathlib import Path
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
import librosa

re_quantize = True
use_quantized = True
# Number of inferences for comparing timings
num_inferences = 1


def predict(processor, model, y, sr):
    # load dummy dataset and read audio files
    input_features = processor(y, sampling_rate=sr, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features, max_length=448)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


@profile()
def main():
    # Configure base model and save directory for compressed model
    model_id = "openai/whisper-large-v2"
    save_dir = "whisper-large-v2"
    inference_file = "audio30.mp3"

    y, sr = librosa.load(inference_file, sr=16000)

    # Export model in ONNX
    if re_quantize:
        model_onnx = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
        model_dir = model_onnx.model_save_dir

    if re_quantize:
        # Run quantization for all ONNX files of exported model
        onnx_models = list(Path(model_dir).glob("*.onnx"))
        print(onnx_models)
        quantizers = [ORTQuantizer.from_pretrained(model_dir, file_name=onnx_model) for onnx_model in onnx_models]

        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

        for quantizer in quantizers:
            # Apply dynamic quantization and save the resulting model
            quantizer.quantize(save_dir=save_dir, quantization_config=qconfig)

    save_dir = "whisper-large-v2"

    if not use_quantized:
        # Create pipeline based on simple ONNX model
        model_dir = "/var/folders/d_/_b8jmlmn4qq8v28y5gj572600000gn/T/tmpi8r1xspk/"
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(model_dir, export=False)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

        # Measure inference of simple onnx model
        start_onnx = datetime.now()
        for i in range(num_inferences):
            res_onnx = predict(processor, model, y, sr)
        end_onnx = datetime.now()

        onnx_inference_time = (end_onnx - start_onnx).total_seconds() / num_inferences
        print(f"Onnx inference time: {onnx_inference_time}")
        print(res_onnx)

    else:
        # Create pipeline based on quantized ONNX model
        # model_quant = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir)
        # tokenizer_quant = AutoTokenizer.from_pretrained(save_dir)
        # feature_extractor_quant = AutoFeatureExtractor.from_pretrained(save_dir)
        # cls_pipeline_quant = pipeline("automatic-speech-recognition", model=model_quant, tokenizer=tokenizer_quant,
        #                               feature_extractor=feature_extractor_quant)

        processor = WhisperProcessor.from_pretrained(save_dir)
        model = ORTModelForSpeechSeq2Seq.from_pretrained(save_dir, export=False)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

        # Measure inference of quantized model
        start_quantized = datetime.now()
        for i in range(num_inferences):
            res_quant = predict(processor, model, y, sr)
        end_quantized = datetime.now()

        quantized_inference_time = (end_quantized - start_quantized).total_seconds() / num_inferences
        print(f"Quantized inference time: {quantized_inference_time}")
        print(res_quant)

    # original or onnx only 19, quantized 13, 2 iters


if __name__ == "__main__":
    main()
