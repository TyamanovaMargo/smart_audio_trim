import os
import json
from pathlib import Path
from pydub import AudioSegment
import whisper
import numpy as np
from typing import Tuple, List, Optional

class SmartAudioTrimmer:
    def __init__(self, input_folder: str, output_folder: str,
                 min_duration: int = 60, max_duration: int = 120,
                 model_size: str = "base"):
        """
        Args:
            input_folder: Path to input audio files folder
            output_folder: Path to save trimmed audio files
            min_duration: Minimum audio length in seconds
            max_duration: Maximum audio length in seconds
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.output_folder.mkdir(parents=True, exist_ok=True)
        (self.output_folder / "logs").mkdir(exist_ok=True)
        print(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)

    def get_audio_duration(self, audio_path: Path) -> float:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # in seconds

    def transcribe(self, audio_path: Path) -> List[dict]:
        """
        Transcribe audio with Whisper, return list of segments:
        Each segment is dict with 'start', 'end', 'text'.
        """
        print(f"Transcribing {audio_path.name}")
        result = self.model.transcribe(str(audio_path))
        segments = result["segments"]
        return segments

    def get_cut_time_from_transcript(self, segments: List[dict]) -> float:
        """
        Based on transcript segments, find cut_time so trimmed audio <= max_duration.
        Strategy:
        - Find segment where cumulative duration > max_duration
        - Cut at end of this segment or closest segment end before max_duration
        """
        if not segments:
            return self.max_duration

        for seg in segments:
            if seg["end"] >= self.max_duration:
                return seg["end"]

        # If none exceed max_duration, cut at last segment end or max_duration
        return segments[-1]["end"] if segments else self.max_duration

    def trim_audio(self, audio_path: Path, cut_time: float) -> Path:
        """
        Trim audio file from start to cut_time seconds.
        Saves trimmed file in output folder preserving folder structure.
        """
        print(f"Trimming {audio_path.name} to {cut_time:.2f} seconds")
        audio = AudioSegment.from_file(audio_path)
        trimmed_audio = audio[:int(cut_time * 1000)]
        rel_path = audio_path.relative_to(self.input_folder)
        output_path = self.output_folder / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trimmed_audio.export(output_path, format=audio_path.suffix[1:])
        return output_path

    def process_pair(self, original_file: Path, diarized_file: Path):
        """
        Process a pair of original and diarized files,
        trimming both to the same duration based on diarized transcription.
        """
        print(f"Processing pair:\n  Original: {original_file.name}\n  Diarized: {diarized_file.name}")
        # Transcribe diarized file to get segments for better speech boundaries
        diarized_segments = self.transcribe(diarized_file)
        cut_time = self.get_cut_time_from_transcript(diarized_segments)

        # Enforce minimum and maximum duration limits
        if cut_time < self.min_duration:
            print(f"Cut time {cut_time:.2f}s less than min_duration {self.min_duration}s, adjusting.")
            cut_time = self.min_duration
        if cut_time > self.max_duration:
            print(f"Cut time {cut_time:.2f}s greater than max_duration {self.max_duration}s, adjusting.")
            cut_time = self.max_duration

        # Trim both files identically
        trimmed_original = self.trim_audio(original_file, cut_time)
        trimmed_diarized = self.trim_audio(diarized_file, cut_time)

        # Save logs
        log_data = {
            "original_file": str(original_file),
            "diarized_file": str(diarized_file),
            "cut_time": cut_time,
            "trimmed_original": str(trimmed_original),
            "trimmed_diarized": str(trimmed_diarized),
        }
        log_path = self.output_folder / "logs" / f"{original_file.stem}_trim_log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Saved log to {log_path}")

    def process_folder(self):
        files = list(self.input_folder.glob("*.*"))
        processed_stems = set()

        for file in files:
            fname = file.name
            # Файл с суффиксом "_original"
            if "_original" in fname:
                base_name = fname.replace("_original", "")
                diarized_name = base_name + "_part1" + file.suffix  # подстройте под свой суффикс
                diarized_path = self.input_folder / diarized_name

                if diarized_path.exists():
                    self.process_pair(file, diarized_path)
                    processed_stems.add(fname)
                    processed_stems.add(diarized_name)
                else:
                    # Без пары - обрезаем только оригинал
                    segments = self.transcribe(file)
                    cut_time = self.get_cut_time_from_transcript(segments)
                    cut_time = max(self.min_duration, min(cut_time, self.max_duration))
                    self.trim_audio(file, cut_time)
                    processed_stems.add(fname)

            elif "_part1" in fname:
                original_name = fname.replace("_part1", "_original")
                original_path = self.input_folder / original_name

                if not original_path.exists() and fname not in processed_stems:
                    # Без пары - обрезаем только диаризованный файл
                    segments = self.transcribe(file)
                    cut_time = self.get_cut_time_from_transcript(segments)
                    cut_time = max(self.min_duration, min(cut_time, self.max_duration))
                    self.trim_audio(file, cut_time)
                    processed_stems.add(fname)



def main():
    INPUT_FOLDER = "input_audio"
    OUTPUT_FOLDER = "output_audio"
    MIN_DURATION = 60  # 1 minute
    MAX_DURATION = 120 # 2 minutes
    MODEL_SIZE = "base"

    trimmer = SmartAudioTrimmer(INPUT_FOLDER, OUTPUT_FOLDER, MIN_DURATION, MAX_DURATION, MODEL_SIZE)
    trimmer.process_folder()

if __name__ == "__main__":
    main()
