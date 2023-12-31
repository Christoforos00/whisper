from datetime import datetime

from faster_whisper import WhisperModel
from memory_profiler import profile

@profile()
def main():
    inference_file = "audio30.mp3"
    model_size = "large-v2"
    num_inferences = 1
    beam_size = 5

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    start_fast = datetime.now()
    if num_inferences == 1:
        segments, info = model.transcribe(inference_file, beam_size=beam_size)
        segments = list(segments)
    else:
        for i in range(num_inferences):
            segments, info = model.transcribe(inference_file, beam_size=beam_size)
            segments = list(segments)
    end_fast = datetime.now()

    fast_inference_time = (end_fast - start_fast).total_seconds() / num_inferences
    print(f"Fast inference time: {fast_inference_time}")
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    # faster is 24 secs with beam_size=5, 13 secs with beam_size=1, fp32
    # faster is 20 secs with beam_size=5, 17 secs with beam_size=1, int8


    # Measure inference of original model
    # cls_pipeline_original = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2")
    #
    # start_original = datetime.now()
    # for i in range(num_inferences):
    #     res_orig = cls_pipeline_original(inference_file)
    # end_original = datetime.now()
    #
    # original_inference_time = (end_original - start_original).total_seconds() / num_inferences
    # print(f"Original inference time: {original_inference_time}")
    # print(res_orig)


if __name__ == "__main__":
    main()