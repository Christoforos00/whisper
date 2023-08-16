"""WhisperCC binding module"""
import time
from datetime import datetime
from pathlib import Path

from memory_profiler import profile
from whispercpp import Whisper
from audio_tools import NdArray, convert_to_float_array, decode_audio

class WhisperTranscriber:
    """WhisperCC transcriber"""

    def __init__(self, model: str) -> None:
        """Initialize the transcriber"""
        self.whisper = Whisper.from_pretrained(model)

    def transcribe_audio(self, audio_file: Path) -> str:
        """Transcribe audio from a file"""
        audio_data = self._load_audio(audio_file)
        start_time = time.time()
        transcription = self.whisper.transcribe(audio_data)
        end_time = time.time()
        execution_time = end_time - start_time
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            "Transcription elapsed execution time:"
            f" {int(hours)}:{int(minutes):02d}:{seconds:06.3f}"
        )
        return transcription

    def _load_audio(self, audio_file: Path) -> NdArray:
        """Load audio from a file"""
        audio_data = decode_audio(audio_file)
        audio_array = convert_to_float_array(audio_data)
        return audio_array

@profile
def main():
    num_inferences = 4
    model = "large"
    # model = "/Users/christof/Downloads/ggml-large.bin"
    whisper_transcriber = WhisperTranscriber(model)
    audio_file = Path("audio30.mp3")

    start = datetime.now()
    for i in range(num_inferences):
        res = whisper_transcriber.transcribe_audio(audio_file)
    end = datetime.now()

    inference_time = (end - start).total_seconds() / num_inferences
    print(f"Inference time: {inference_time}")
    print(res)


if __name__ == "__main__":
    main()
