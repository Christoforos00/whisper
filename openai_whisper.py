from datetime import datetime

import whisper

inference_file = "audio30.mp3"
model_size = "large-v2"
num_inferences = 4

model = whisper.load_model(model_size)

start_openai = datetime.now()
for i in range(num_inferences):
    result = model.transcribe(inference_file, beam_size=1)
end_openai = datetime.now()

print(result["text"])
openai_inference_time = (end_openai - start_openai).total_seconds() / num_inferences
print(f"Fast inference time: {openai_inference_time}")

# print(result["text"])
# openai whisper implementation is 60 secs with beam_size=5, 20 secs with beam_size=1, fp32
