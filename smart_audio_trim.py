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

    def get_speech_duration(self, segments: List[dict], end_time: float) -> float:
        """
        Calculate total speech duration up to end_time
        """
        total_speech = 0.0
        for seg in segments:
            seg_start = seg["start"]
            seg_end = min(seg["end"], end_time)
            if seg_start < end_time:
                total_speech += (seg_end - seg_start)
        return total_speech

    def find_cut_time_for_speech_duration(self, segments: List[dict], 
                                          target_speech_duration: float) -> float:
        """
        Find the cut time needed to get approximately target_speech_duration of actual speech
        """
        if not segments:
            return self.max_duration
        
        cumulative_speech = 0.0
        for seg in segments:
            seg_duration = seg["end"] - seg["start"]
            if cumulative_speech + seg_duration >= target_speech_duration:
                # Cut at the end of this segment
                return seg["end"]
            cumulative_speech += seg_duration
        
        # If we haven't reached target, return last segment end
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
        trimming both to have approximately the same amount of actual speech.
        """
        print(f"Processing pair:\n  Original: {original_file.name}\n  Diarized: {diarized_file.name}")
        
        # Transcribe both files
        print("Transcribing diarized file...")
        diarized_segments = self.transcribe(diarized_file)
        print("Transcribing original file...")
        original_segments = self.transcribe(original_file)
        
        # Calculate target speech duration based on diarized file
        # (assuming diarized has cleaner speech)
        diarized_speech_duration = sum(seg["end"] - seg["start"] for seg in diarized_segments)
        target_speech_duration = min(diarized_speech_duration, self.max_duration)
        target_speech_duration = max(target_speech_duration, self.min_duration)
        
        print(f"Target speech duration: {target_speech_duration:.2f} seconds")
        
        # Find cut times for each file to get similar speech duration
        diarized_cut_time = self.find_cut_time_for_speech_duration(diarized_segments, target_speech_duration)
        original_cut_time = self.find_cut_time_for_speech_duration(original_segments, target_speech_duration)
        
        # Enforce maximum duration limits
        diarized_cut_time = min(diarized_cut_time, self.max_duration)
        original_cut_time = min(original_cut_time, self.max_duration)
        
        print(f"Diarized cut time: {diarized_cut_time:.2f}s")
        print(f"Original cut time: {original_cut_time:.2f}s")
        
        # Calculate actual speech in trimmed versions
        diarized_actual_speech = self.get_speech_duration(diarized_segments, diarized_cut_time)
        original_actual_speech = self.get_speech_duration(original_segments, original_cut_time)
        
        print(f"Diarized speech duration: {diarized_actual_speech:.2f}s")
        print(f"Original speech duration: {original_actual_speech:.2f}s")
        
        # Trim both files with their respective cut times
        trimmed_original = self.trim_audio(original_file, original_cut_time)
        trimmed_diarized = self.trim_audio(diarized_file, diarized_cut_time)

        # Save logs
        log_data = {
            "original_file": str(original_file),
            "diarized_file": str(diarized_file),
            "original_cut_time": original_cut_time,
            "diarized_cut_time": diarized_cut_time,
            "original_speech_duration": original_actual_speech,
            "diarized_speech_duration": diarized_actual_speech,
            "target_speech_duration": target_speech_duration,
            "trimmed_original": str(trimmed_original),
            "trimmed_diarized": str(trimmed_diarized),
        }
        log_path = self.output_folder / "logs" / f"{original_file.stem}_trim_log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Saved log to {log_path}")
        print("-" * 50)

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
                    speech_duration = sum(seg["end"] - seg["start"] for seg in segments)
                    target_speech = min(speech_duration, self.max_duration)
                    target_speech = max(target_speech, self.min_duration)
                    cut_time = self.find_cut_time_for_speech_duration(segments, target_speech)
                    cut_time = min(cut_time, self.max_duration)
                    self.trim_audio(file, cut_time)
                    processed_stems.add(fname)

            elif "_part1" in fname:
                original_name = fname.replace("_part1", "_original")
                original_path = self.input_folder / original_name

                if not original_path.exists() and fname not in processed_stems:
                    # Без пары - обрезаем только диаризованный файл
                    segments = self.transcribe(file)
                    speech_duration = sum(seg["end"] - seg["start"] for seg in segments)
                    target_speech = min(speech_duration, self.max_duration)
                    target_speech = max(target_speech, self.min_duration)
                    cut_time = self.find_cut_time_for_speech_duration(segments, target_speech)
                    cut_time = min(cut_time, self.max_duration)
                    self.trim_audio(file, cut_time)
                    processed_stems.add(fname)



def main():
    INPUT_FOLDER = "/Users/margotiamanova/Desktop/PROJECTS/smart_audio_trim/input_audio_test"
    OUTPUT_FOLDER = "/Users/margotiamanova/Desktop/PROJECTS/smart_audio_trim/output_audio_test"
    MIN_DURATION = 60  # 1 minute
    MAX_DURATION = 120 # 2 minutes
    MODEL_SIZE = "base"

    trimmer = SmartAudioTrimmer(INPUT_FOLDER, OUTPUT_FOLDER, MIN_DURATION, MAX_DURATION, MODEL_SIZE)
    trimmer.process_folder()

if __name__ == "__main__":
    main()