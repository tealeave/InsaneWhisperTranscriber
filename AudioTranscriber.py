import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import subprocess
import shutil
import json

class AudioTranscriber:
    """
    A class for transcribing audio files to text using OpenAI's Whisper model.

    Initializes the transcription model and provides a method to transcribe an audio file
    into a text file, including the total time taken for transcription.

    Attributes:
        model_id (str, optional): Identifier for the Whisper model. Defaults to "openai/whisper-large-v3".

    Methods:
        transcribe(audio_file, output_file_name): Transcribes the given audio file.
            Saves the transcription to the specified output file.

    Usage:
        transcriber = AudioTranscriber(model_id="optional_model_id")
        transcriber.transcribe("path/to/audio_file.wav", "output_file_name.txt")

    The output text file includes the transcription of the audio file and the total time taken for transcription.
    """

    def __init__(self, model_id="openai/whisper-large-v3"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def format_timestamp(self, seconds, format_type="sec"):
        """Converts seconds to the specified format ('sec' or 'min_sec')."""
        if seconds is None:
            return "Unknown"

        if format_type == "min_sec":
            mins = int(seconds // 60)
            secs = float(seconds % 60)
            return f"{mins}:{secs:05.2f}"

        return f"{seconds:.2f}"


    def transcribe(self, audio_file, output_file_name, timestamp_format="sec"):
        """
        Transcribes the given audio file and saves the transcription to the specified output file.

        Parameters:
            audio_file (str): The path to the audio file to be transcribed.
            output_file_name (str): The path where the transcription will be saved.
            timestamp_format (str, optional): The format of timestamps in the transcription.
                "sec" for seconds (default) or "min_sec" for minutes and seconds.

        The method writes the transcription to the output file with timestamps for each spoken segment.
        The transcription also includes the total time taken for the process, appended at the end of the file.
        """
        start_time = time.time()
        result = self.pipe(audio_file, generate_kwargs={"language": "english"})

        with open(output_file_name, "w") as f:
            for chunk in result["chunks"]:
                start, end = chunk["timestamp"]
                formatted_start = self.format_timestamp(start, timestamp_format)
                formatted_end = self.format_timestamp(end, timestamp_format)
                f.write(f'({formatted_start}, {formatted_end}): {chunk["text"]}\n')

        total_time = time.time() - start_time
        with open(output_file_name, "a") as f:
            f.write(f'\nTotal time taken: {total_time} seconds')

    def insanely_fast_whisper(self, audio_file, output_dir, temp_file, processed_dir, timestamp_format, hf_token, batch_size):
        """
        Processes an audio file: transcribes it, saves the transcription, and moves the file to the processed directory.

        Parameters:
            audio_file (Path): The path to the audio file to be processed.
            output_dir (Path): The directory where the transcription will be saved.
            temp_dir (Path): The directory for temporary files.
            processed_dir (Path): The directory where the processed audio files will be moved.
            timestamp_format (str): The format of timestamps in the transcription ('sec' or 'min_sec').
            hf_token (str): The Hugging Face API token for authentication.
        """
        start_time = time.time()
        print(f'Transcribing {audio_file}')

        command = [
            "insanely-fast-whisper",
            "--file-name", str(audio_file),
            "--hf_token", hf_token,
            "--model", "openai/whisper-medium",
            "--batch-size", batch_size,
            "--transcript-path", str(temp_file),
        ]
        subprocess.run(command)

        output_file = output_dir.joinpath(audio_file.stem + ".txt")

        with open(temp_file, 'r') as file:
            data = json.load(file)

        with open(output_file, 'w') as file:
            for entry in data['speakers']:
                speaker = entry['speaker']
                start, end = entry['timestamp']
                formatted_start = self.format_timestamp(start, timestamp_format)
                formatted_end = self.format_timestamp(end, timestamp_format)
                text = entry['text']
                file.write(f"{speaker} [{formatted_start}-{formatted_end}]: {text}\n")

        total_time = time.time() - start_time
        with open(output_file, "a") as file:
            file.write(f"Total processing time: {total_time:.2f} seconds\n")

        temp_file.unlink()
        shutil.move(str(audio_file), str(processed_dir))

        print(f"Transcript written to {output_file}")
        print(f"Processed data moved {audio_file} to {processed_dir}")

            # Clearing the GPU cache after processing each file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


