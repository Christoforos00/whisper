from datetime import datetime

import whisper
from memory_profiler import profile


@profile()
def main():
    inference_file = "audio30.mp3"
    model_size = "large-v2"
    num_inferences = 1
    beam_size = 5

    model = whisper.load_model(model_size)

    start_openai = datetime.now()
    if num_inferences == 1:
        result = model.transcribe(inference_file, beam_size=beam_size)
    else:
        for i in range(num_inferences):
            result = model.transcribe(inference_file, beam_size=beam_size)
    end_openai = datetime.now()

    print(result["text"])
    openai_inference_time = (end_openai - start_openai).total_seconds() / num_inferences
    print(f"Fast inference time: {openai_inference_time}")

    # print(result["text"])
    # openai whisper implementation is 60 secs with beam_size=5, 20 secs with beam_size=1, fp32


if __name__ == "__main__":
    main()
