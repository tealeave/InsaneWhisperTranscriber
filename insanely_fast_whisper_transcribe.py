"""


This script is designed for the automated transcription of audio files using the AudioTranscriber class,
which utilizes OpenAI's Whisper model. The script processes all audio files located in a specified directory,
transcribes them, and saves the results in another directory.

The script sets up necessary directories for input audio files, processed files, and transcription results. 
It ensures that these directories exist before proceeding with the transcription process. The transcription 
includes timestamps and can be formatted as per the user's preference.


Requirement:
    Follow the installation guide on https://github.com/Vaibhavs10/insanely-fast-whisper/tree/main
    hf_token can be obtained from https://huggingface.co/pyannote/speaker-diarization-3.1, accept usr conditions and create a huggingface token

Attributes:
    WORK_DIR (Path): The directory where the script is located. This is used as the base for other paths.
    AUDIO_FILES_DIR (Path): The directory where the audio files to be transcribed are stored.
    RESULTS_DIR (Path): The directory where transcription results are saved.
    TEMP_FILE (Path): The path for temporary files generated during transcription.
    PROCESSED_DIR (Path): The directory where audio files are moved after transcription.
    BATCH_SIZE (str): The batch size for processing, can be adjusted based on available resources.
    timestamp_format (str): The format of timestamps in the transcription output ('sec' or 'min_sec').

Usage:
    Ensure that the AudioTranscriber class is properly defined and available.
    Place audio files in the 'audio_files' directory relative to the script's location.
    Run the script. Transcribed files will be saved in the 'results' directory, 
    and processed audio files will be moved to the 'done' subdirectory within 'audio_files'.

Note:
    The batch size can be adjusted based on the GPU and power available, especially when using a GPU.
"""

from pathlib import Path
from AudioTranscriber import AudioTranscriber
import constants

# Set up directories and configurations
WORK_DIR = Path(__file__).parent.resolve()
AUDIO_FILES_DIR = WORK_DIR.joinpath('audio_files')
RESULTS_DIR = WORK_DIR.joinpath("results")
TEMP_FILE = WORK_DIR.joinpath("output.json")
PROCESSED_DIR = AUDIO_FILES_DIR.joinpath('done')
BATCH_SIZE = "16" # 24 or higher if GPU and power can take it
timestamp_format = "min_sec"

# Create directories if they do not exist
AUDIO_FILES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the transcriber
transcriber = AudioTranscriber()

# Process each audio file
for audio_file in AUDIO_FILES_DIR.iterdir():
    if audio_file.is_file():
        transcriber.insanely_fast_whisper(  
                                            audio_file,
                                            RESULTS_DIR,
                                            TEMP_FILE,
                                            PROCESSED_DIR,
                                            timestamp_format,
                                            constants.PYANNOTE_ACCESS_TOKEN,
                                            BATCH_SIZE
                                            )
