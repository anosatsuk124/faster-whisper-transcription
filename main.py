from faster_whisper import WhisperModel
import sys

model_size = "large-v3"

audio_file = sys.argv[1]

# Run on GPU with FP16
model = WhisperModel(model_size, device="cpu", compute_type="int8")

transcription = ""

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe(audio_file, beam_size=5, language="ja")

# print("Detected language '%s' with probability %f" %
# (info.language, info.language_probability))


with open(f"{audio_file}.txt", "w") as f:
    for segment in segments:
        s = "[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text)
        print(s)
        f.write(s + "\n")

        transcription += segment.text

    print()

    print(transcription)
    f.write(transcription + "\n")
