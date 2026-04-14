import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from wav import read_wav, play_pcm_stream

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    interface = "eth0"
    robot_ip = "192.168.123.164"
    domain_id = 0
    wav_file = "recorded_audio.wav"

    if len(sys.argv) > 1:
        wav_file = sys.argv[1]

    logger.info(f"Init channel via {interface} to {robot_ip}")
    ChannelFactoryInitialize(domain_id, interface)
    audio = AudioClient()
    audio.SetTimeout(10.0)
    audio.Init()

    pcm_list, sample_rate, num_channels, is_ok = read_wav(wav_file)
    if not is_ok:
        logger.error("Не вдалося прочитати WAV")
        return

    logger.info(f"WAV: rate={sample_rate}, channels={num_channels}, bytes={len(pcm_list)}")

    if sample_rate != 16000 or num_channels != 1:
        logger.error("Потрібен WAV 16kHz mono")
        return

    try:
        play_pcm_stream(
            audio,
            pcm_list,
            stream_name="example",
            chunk_size=16000,
            sleep_time=0.4,
            logger=logger
        )
        logger.info("Playback finished")
    finally:
        audio.PlayStop("example")

if __name__ == "__main__":
    main()
