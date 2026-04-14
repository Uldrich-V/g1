import os
import cv2
import time
import wave
import base64
import audioop
import signal
import asyncio
import threading
import subprocess
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

from unitree_webrtc_connect.webrtc_driver import UnitreeWebRTCConnection
from unitree_webrtc_connect.constants import WebRTCConnectionMethod

from wav import read_wav_from_bytes, play_pcm_stream


# =========================
# CONFIG
# =========================

NETWORK_INTERFACE = "eth0"

STT_MODEL = "gpt-4o-mini-transcribe"
LLM_MODEL = "gpt-4.1-mini"
VISION_MODEL = "gpt-4.1"

USE_CAMERA_WINDOW = True   # True = show cv2 window
CAMERA_ANALYZE_MAX_WIDTH = 960

SYSTEM_PROMPT = (
    "You are a voice assistant of the Unitree G1 robot. "
    "You MUST strictly use ONLY these languages: Czech, Ukrainian, or English. "
    "Never use any other language. Never use Russian or Belarusian. "
    "Primary language is Czech. "
    "If the user speaks Czech, respond in Czech. "
    "If the user speaks Ukrainian, respond in Ukrainian. "
    "If the user speaks English, respond in English. "
    "If uncertain between Czech and another allowed language, prefer Czech. "
    "Never mix languages. Never switch languages unless the user does. "
    "Always answer directly, clearly, and confidently. "
    "Do not ask unnecessary clarifying questions. "
    "Do not repeat the user's question. "
    "Keep answers short, natural, useful, and polite. "
    "If the request is unsafe, illegal, disallowed, or harmful, politely refuse in the same language as the user. "
    "Examples of refusal style: "
    "'Bohužel s tímto nemohu pomoci.' "
    "'На жаль, я не можу з цим допомогти.' "
    "'Sorry, I cannot help with that.'"
)


# =========================
# GLOBAL CAMERA STATE
# =========================

latest_frame = None
latest_frame_lock = threading.Lock()
camera_ready = threading.Event()
camera_stop_event = threading.Event()


# =========================
# AUDIO HELPERS
# =========================

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


def record_from_robot_to_file(output_path: str, interface: str = "eth0", seconds: float = 1.5) -> str:
    """
    Uses existing sitport/audio_rec.py recorder.
    Assumes this script lives in a folder that also has subfolder: ./sitport/audio_rec.py
    """
    base = Path(__file__).parent
    rec_dir = base / "sitport"
    rec_file = rec_dir / "recorded_audio.wav"

    if not rec_dir.exists():
        raise RuntimeError(f"Missing folder: {rec_dir}")
    if not (rec_dir / "audio_rec.py").exists():
        raise RuntimeError(f"Missing file: {rec_dir / 'audio_rec.py'}")

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
    chunk_seconds: float = 1.5,
    threshold: int = 200,
    prebuffer_chunks: int = 2,
    max_silence_chunks: int = 3,
    max_total_chunks: int = 15,
) -> str:
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


# =========================
# OPENAI HELPERS
# =========================

def transcribe_audio(client: OpenAI, filename: str) -> str:
    print("[STT] Transcribing...")
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            prompt=(
                "The speaker uses only Czech, Ukrainian, or English. "
                "Prefer Czech if unclear. "
                "Do not output Russian or Belarusian."
            ),
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


def speak_robot(audio_client: AudioClient, client: OpenAI, text: str) -> None:
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
        return

    duration = len(pcm_list) / (sample_rate * num_channels * 2)
    print(f"[ROBOT] Playing (~{duration:.1f}s)...")
    play_pcm_stream(audio_client, pcm_list, "tts_reply")
    time.sleep(duration + 0.4)
    audio_client.PlayStop("tts_reply")


# =========================
# CAMERA / VISION HELPERS
# =========================

def get_latest_frame_copy():
    global latest_frame
    with latest_frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()


def resize_for_vision(img):
    h, w = img.shape[:2]
    if w <= CAMERA_ANALYZE_MAX_WIDTH:
        return img
    scale = CAMERA_ANALYZE_MAX_WIDTH / float(w)
    nh = int(h * scale)
    return cv2.resize(img, (CAMERA_ANALYZE_MAX_WIDTH, nh))


def frame_to_base64_jpg(img) -> str:
    img = resize_for_vision(img)
    ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def looks_like_vision_question(text: str) -> bool:
    t = text.lower()

    triggers = [
        "що ти бачиш",
        "що переді мною",
        "що навколо",
        "подивись",
        "опиши що бачиш",
        "what do you see",
        "what is in front of me",
        "look around",
        "describe what you see",
        "co vidíš",
        "co je přede mnou",
        "podívej se",
        "popiš co vidíš",
    ]
    return any(x in t for x in triggers)


def analyze_image_with_question(client: OpenAI, img, user_question: str) -> str:
    image_b64 = frame_to_base64_jpg(img)

    response = client.responses.create(
        model=VISION_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_question},
                    {"type": "input_image", "image_base64": image_b64},
                ],
            },
        ],
    )
    return response.output_text.strip()


async def video_callback(track):
    global latest_frame

    print("[CAM] Video track received")
    camera_ready.set()

    while not camera_stop_event.is_set():
        try:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")

            with latest_frame_lock:
                latest_frame = img.copy()

            if USE_CAMERA_WINDOW:
                cv2.imshow("G1 Camera", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[CAM] Window closed by user")
                    camera_stop_event.set()
                    break

        except Exception as e:
            print(f"[CAM] Video error: {e}")
            break

    cv2.destroyAllWindows()


async def camera_main():
    conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalAP)

    await conn.connect()
    print("[CAM] Connected! Enabling video...")

    conn.video.add_track_callback(video_callback)
    conn.video.switchVideoChannel(True)

    while not camera_stop_event.is_set():
        await asyncio.sleep(0.2)

    print("[CAM] Stopping camera task")


def start_camera_thread():
    def runner():
        try:
            asyncio.run(camera_main())
        except Exception as e:
            print(f"[CAM] Camera thread failed: {e}")

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t


# =========================
# OPTIONAL COMMANDS
# =========================

def detect_simple_command(text: str) -> Optional[str]:
    t = text.lower()

    if any(x in t for x in ["stop camera", "turn off camera", "vypni kameru", "вимкни камеру"]):
        return "stop_camera"

    if any(x in t for x in ["show camera", "start camera", "zapni kameru", "увімкни камеру"]):
        return "start_camera"

    return None


# =========================
# MAIN
# =========================

def main() -> None:
    load_dotenv("/home/username/old_project/.env")
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

    print("[INIT] Starting camera thread...")
    camera_thread = start_camera_thread()

    history: List[Dict] = []

    print("=== Unitree G1 Voice + Vision Assistant ===")
    print("Press Enter to start recording. Ctrl+C to exit.")
    print("Camera starts in background.")

    try:
        while True:
            input("\n[READY] Press Enter to speak...")

            print("[REC] Recording — speak now, stops on silence...")
            wav_path = listen_until_silence(interface=NETWORK_INTERFACE)

            if not file_has_audio(wav_path):
                print("[INFO] No audio captured, try again.")
                continue

            user_text = transcribe_audio(client, wav_path)

            if not user_text:
                print("[INFO] Nothing recognized, try again.")
                continue

            cmd = detect_simple_command(user_text)
            if cmd == "stop_camera":
                camera_stop_event.set()
                reply = ask_llm(client, history, user_text)
                speak_robot(audio_client, client, reply)
                continue

            if looks_like_vision_question(user_text):
                frame = get_latest_frame_copy()
                if frame is None:
                    reply = (
                        "Камера ще не готова." if "що" in user_text.lower()
                        else "Kamera ještě není připravena." if "co" in user_text.lower() or "vidíš" in user_text.lower()
                        else "The camera is not ready yet."
                    )
                    print(f"[Bot] {reply}")
                    speak_robot(audio_client, client, reply)
                    continue

                print("[VISION] Analyzing current camera frame...")
                try:
                    reply_text = analyze_image_with_question(client, frame, user_text)
                    history.append({"role": "user", "content": user_text})
                    history.append({"role": "assistant", "content": reply_text})
                    history[:] = history[-20:]
                    print(f"[Bot] {reply_text}")
                    speak_robot(audio_client, client, reply_text)
                except Exception as e:
                    print(f"[VISION ERROR] {e}")
                    fallback = (
                        "На жаль, я не зміг проаналізувати зображення."
                        if "що" in user_text.lower()
                        else "Bohužel se mi nepodařilo analyzovat obraz."
                        if "co" in user_text.lower() or "vidíš" in user_text.lower()
                        else "Sorry, I could not analyze the image."
                    )
                    speak_robot(audio_client, client, fallback)
                continue

            reply_text = ask_llm(client, history, user_text)
            speak_robot(audio_client, client, reply_text)

    except KeyboardInterrupt:
        print("\n[STOP] Stopped by user.")
    finally:
        camera_stop_event.set()
        time.sleep(1.0)
        cv2.destroyAllWindows()
        if camera_thread.is_alive():
            print("[EXIT] Camera thread asked to stop.")


if __name__ == "__main__":
    main()
