import os
import subprocess
import time
import wave
import audioop
import signal
from collections import deque
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from wav import read_wav_from_bytes, play_pcm_stream


NETWORK_INTERFACE = "eth0"

STT_MODEL = "gpt-4o-mini-transcribe"
LLM_MODEL = "gpt-4.1-mini"

WAKE_WORDS = [
    "hello robot",                   # English
    "hola robot", "hola, robot",     # Spanish
    "привіт робот",                  # Ukrainian
    "ahoj robote", "ahoj robot",     # Czech
    "bonjour robot",                 # French
    "hallo robot",                   # German
]

SYSTEM_PROMPT = (
    "You are a voice assistant of the Unitree G1 robot. "
    "You support any language — including English, Ukrainian, Czech, Spanish, German, French, and others. "
    "Always detect the language the user is speaking and reply in that same language. "
    "Keep your answers short, natural, and polite. "
    "Never give overly long responses. "
    "If the question is unclear, ask a brief clarifying question first."
)


# ── Audio helpers ────────────────────────────────────────────────────────────

def file_has_audio(filename: str, min_size_bytes: int = 4000) -> bool:
    path = Path(filename)
    return path.exists() and path.stat().st_size >= min_size_bytes


def get_wav_rms(filename: str) -> int:
    with wave.open(filename, "rb") as w:
        data = w.readframes(w.getnframes())
        if not data:
            return 0
        return audioop.rms(data, 2)


def merge_wavs(inputs: List[str], output: str) -> str:
    if not inputs:
        raise RuntimeError("No WAV files to merge")

    params = None
    all_frames = []

    for path in inputs:
        with wave.open(path, "rb") as w:
            if params is None:
                params = w.getparams()
            else:
                if (
                    w.getnchannels() != params.nchannels
                    or w.getsampwidth() != params.sampwidth
                    or w.getframerate() != params.framerate
                ):
                    raise RuntimeError("WAV files have different parameters")
            all_frames.append(w.readframes(w.getnframes()))

    with wave.open(output, "wb") as out:
        out.setparams(params)
        for frames in all_frames:
            out.writeframes(frames)

    return output


def record_from_robot_to_file(output_path: str, interface: str = "eth0", seconds: int = 1) -> str:
    base = Path(__file__).parent
    rec_dir = base / "sitport"
    rec_file = rec_dir / "recorded_audio.wav"

    if rec_file.exists():
        rec_file.unlink()

    proc = subprocess.Popen(
        ["python3", "audio_rec.py", interface],
        cwd=rec_dir
    )

    try:
        time.sleep(seconds)
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=3)
    except Exception:
        proc.kill()
        proc.wait()

    if not rec_file.exists():
        raise RuntimeError("audio_rec.py did not create recorded_audio.wav")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    rec_file.replace(out)
    return str(out)


def listen_until_silence(
    interface: str = "eth0",
    chunk_seconds: float = 1.0,
    threshold: int = 500,
    prebuffer_chunks: int = 2,
    max_silence_chunks: int = 2,
    max_total_chunks: int = 15,
) -> str:
    """
    Records 1-second chunks, waits for voice to start,
    keeps a pre-buffer before voice, stops after silence.
    Returns path to merged WAV.
    """
    base = Path(__file__).parent
    tmp_dir = base / "tmp_vad"
    tmp_dir.mkdir(exist_ok=True)

    prebuffer: deque = deque(maxlen=prebuffer_chunks)
    collected: List[str] = []
    voice_started = False
    silence_count = 0

    for i in range(max_total_chunks):
        chunk_path = tmp_dir / f"chunk_{i}.wav"
        record_from_robot_to_file(str(chunk_path), interface=interface, seconds=chunk_seconds)

        rms = get_wav_rms(str(chunk_path))
        print(f"[VAD] chunk={i}, rms={rms}")

        if not voice_started:
            prebuffer.append(str(chunk_path))
            if rms > threshold:
                print("[VAD] Voice started")
                voice_started = True
                collected.extend(list(prebuffer))
                silence_count = 0
            continue

        collected.append(str(chunk_path))

        if rms > threshold:
            silence_count = 0
        else:
            silence_count += 1
            if silence_count >= max_silence_chunks:
                print("[VAD] Voice ended")
                break

    if not collected:
        raise RuntimeError("No voice detected")

    output_path = tmp_dir / "utterance.wav"
    merge_wavs(collected, str(output_path))
    return str(output_path)


# ── AI helpers ───────────────────────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    text_lower = text.lower().strip()
    return any(w in text_lower for w in WAKE_WORDS) or "robot" in text_lower


def extract_question_from_wake(text: str) -> str:
    """If user said 'hello robot, <question>' extract everything after 'robot'."""
    idx = text.lower().find("robot")
    if idx == -1:
        return ""
    after = text[idx + len("robot"):].strip().lstrip(".,!?-– ").strip()
    return after


def transcribe_audio(client: OpenAI, filename: str) -> str:
    print("[STT] Transcribing...")
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
        )
    text = transcript.text.strip()
    print(f"[You] {text}")
    return text


def ask_llm(client: OpenAI, history: List[Dict], user_text: str) -> str:
    print("[LLM] Thinking...")
    history.append({"role": "user", "content": user_text})
    history[:] = history[-20:]

    input_items = [{"role": "system", "content": SYSTEM_PROMPT}]
    input_items.extend(history)

    response = client.responses.create(
        model=LLM_MODEL,
        input=input_items,
    )

    reply_text = response.output_text.strip()
    history.append({"role": "assistant", "content": reply_text})
    history[:] = history[-20:]

    print(f"[Bot] {reply_text}")
    return reply_text


def speak_robot(audio_client: AudioClient, client: OpenAI, text: str) -> float:
    """Returns estimated playback duration in seconds."""
    print("[TTS] Generating speech via OpenAI...")
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        response_format="wav",
    )

    pcm_list, sample_rate, num_channels, is_ok = read_wav_from_bytes(response.content)

    if not is_ok:
        print("[ERROR] Failed to parse TTS WAV")
        return 0.0

    duration = len(pcm_list) / (sample_rate * num_channels * 2)
    print(f"[ROBOT] Playing ({len(pcm_list)} bytes, {sample_rate}Hz, {num_channels}ch, ~{duration:.1f}s)...")
    play_pcm_stream(audio_client, pcm_list, "tts_reply")
    # Chunks are sent faster than playback — wait for the robot to finish playing
    time.sleep(duration + 0.5)
    audio_client.PlayStop("tts_reply")
    return duration


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    client = OpenAI(api_key=api_key)

    print("[INIT] Initializing Unitree channel...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    print("[INIT] Initializing AudioClient...")
    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    audio_client.SetVolume(100)

    history: List[Dict] = []

    print("=== Unitree G1 Voice GPT (Nazar) ===")
    print("Say 'Hello Robot' to wake me up. Ctrl+C to exit.")

    while True:
        try:
            # === Phase 1: Listen for wake word ===
            print("\n[LISTEN] Waiting for 'Hello Robot'...")
            wav_path = listen_until_silence(interface=NETWORK_INTERFACE)

            if not file_has_audio(wav_path):
                continue

            wake_text = transcribe_audio(client, wav_path)

            if not is_wake_word(wake_text):
                continue

            # Check if the question was already in the wake phrase
            user_text = extract_question_from_wake(wake_text)

            if not user_text:
                # === Phase 2: Listen for the actual question ===
                print("[WAKE] Hello! Listening for your question...")
                wav_path = listen_until_silence(interface=NETWORK_INTERFACE)

                if not file_has_audio(wav_path):
                    continue

                user_text = transcribe_audio(client, wav_path)

            if not user_text:
                continue

            reply_text = ask_llm(client, history, user_text)
            speak_robot(audio_client, client, reply_text)

        except KeyboardInterrupt:
            print("\n[STOP] Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
