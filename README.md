# whisper

In order to propose a device for running Whisper, we need to investigate its inference speed and memory footprint.
Multiple repos promise good results, but benchmarking ourselves locally is essential in order to proceed with specific devices (e.g. IPhone).
Starting with Whisper large (v2), we create demos using different optimized versions available in open-source.
This repo contains code for different flavors of Whisper large (v2):
- HuggingFace version (huggingface_whisper.py)
- HuggingFace version quantized (huggingface_quantized.py)
- OpenAI version (openai_whisper.py)
- faster_whisper version (whisper_fast.py)
- Whisper Cpp version (whispercpp.py)

benchmarks on Mac, using audio30.mp3 for speech to text:

- huggingface whisper is 19 secs with beam_size=?? , fp32
- huggingface whisper quantized is 13 secs with beam_size=??, quint8

- openai implementation is 60 secs with beam_size=5, 20 secs with beam_size=1, fp32

- faster_whisper is 24 secs with beam_size=5, 13 secs with beam_size=1, fp32
- faster_whisper is 20 secs with beam_size=5, 17 secs with beam_size=1, int8

Current thoughts:
- In huggingface version, the quantized model is indeed faster and lighter.
- faster_whisper is indeed faster and lighter than the openai version, as promised.
</br> But:
- Both the openai version and faster_whisper version are slower than huggingface (quantized or not). This needs further investigation

Next steps:
- detailed memory footprint results are missing, would use the "memory-profiler" package
- whisper cpp code is here but hasn't been tested
- investigate why the openai version is slower than huggingface
- run the code on google colab cpu, to check for differences