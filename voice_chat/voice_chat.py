import os
import subprocess
import tempfile
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


RECORD_SECONDS = 5
PULSE_SOURCE = "alsa_input.platform-sound.analog-stereo"
NETWORK_INTERFACE = "eth0"

# 51=DMIC1, 52=DMIC2, 53=DMIC3, 54=DMIC4
DMIC_MUX = 54

STT_MODEL = "gpt-4o-mini-transcribe"
LLM_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = (
    "Ти голосовий асистент робота Unitree G1. "
    "Ти розумієш тільки англійську і відповідаєш нею. "
    "Автоматично визначай мову користувача і відповідай тією ж мовою. "
    "Відповідай коротко, природно, ввічливо. "
    "Не використовуй надто довгі відповіді. "
    "Якщо питання неясне, спочатку коротко уточни."
)


def run_cmd(cmd: List[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Команда не виконалась:\n{' '.join(cmd)}\n\n{result.stderr}"
        )

def has_voice(filename, threshold=500):
    import wave, audioop

    with wave.open(filename, 'rb') as w:
        data = w.readframes(w.getnframes())
        rms = audioop.rms(data, 2)

    return rms > threshold


def record_from_robot(interface="eth0", seconds=5):
    import subprocess
    import time
    from pathlib import Path

    base = Path(__file__).parent
    rec_dir = base / "sitport"
    path = rec_dir / "recorded_audio.wav"

    if path.exists():
        path.unlink()

    print(f"[REC] Запис з робота {seconds} сек...")

    proc = subprocess.Popen(
        ["python3", "audio_rec.py", interface],
        cwd=rec_dir
    )

    time.sleep(seconds)
    proc.send_signal(2)  # Ctrl+C
    time.sleep(1)

    if not path.exists():
        raise RuntimeError("Файл не створився")

    print("[OK] Записано з робота")
    return str(path)
def file_has_audio(filename: str, min_size_bytes: int = 4000) -> bool:
    path = Path(filename)
    return path.exists() and path.stat().st_size >= min_size_bytes


def transcribe_audio(client: OpenAI, filename: str) -> str:
    print("[STT] Розпізнаю мовлення...")
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
        )
    text = transcript.text.strip()
    print(f"[Ти] {text}")
    return text


def ask_llm(client: OpenAI, history: List[Dict], user_text: str) -> str:
    print("[LLM] Думаю над відповіддю...")

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

    print(f"[Бот] {reply_text}")
    return reply_text


def speak_robot(audio_client: AudioClient, text: str) -> None:
    print("[ROBOT] Говорю через динамік робота...")
    code = audio_client.TtsMaker(text, 1)
    print(f"[ROBOT] TtsMaker result: {code}")

def get_wav_rms(filename: str) -> int:
    with wave.open(filename, "rb") as w:
        data = w.readframes(w.getnframes())
        if not data:
            return 0
        return audioop.rms(data, 2)


def merge_wavs(inputs: List[str], output: str) -> str:
    if not inputs:
        raise RuntimeError("Немає WAV файлів для склейки")

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
                    raise RuntimeError("WAV файли мають різні параметри")

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
        raise RuntimeError("audio_rec.py не створив recorded_audio.wav")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        out.unlink()

    rec_file.replace(out)
    return str(out)


def listen_until_silence_with_prebuffer(
    interface: str = "eth0",
    chunk_seconds: float = 1.0,
    threshold: int = 500,
    prebuffer_chunks: int = 2,
    max_silence_chunks: int = 2,
    max_total_chunks: int = 15,
) -> str:
    """
    Чекає голос, зберігає кілька шматків ДО початку мовлення,
    потім пише до тиші і склеює все в один WAV.
    """
    base = Path(__file__).parent
    tmp_dir = base / "tmp_vad"
    tmp_dir.mkdir(exist_ok=True)

    prebuffer = deque(maxlen=prebuffer_chunks)
    collected: List[str] = []

    voice_started = False
    silence_count = 0

    print("[WAIT] Очікую голос...")

    for i in range(max_total_chunks):
        chunk_path = tmp_dir / f"chunk_{i}.wav"

        record_from_robot_to_file(
            output_path=str(chunk_path),
            interface=interface,
            seconds=chunk_seconds,
        )

        rms = get_wav_rms(str(chunk_path))
        print(f"[VAD] chunk={i}, rms={rms}")

        if not voice_started:
            prebuffer.append(str(chunk_path))

            if rms > threshold:
                print("[VOICE] Початок мовлення")
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
                print("[VOICE] Кінець мовлення")
                break

    if not collected:
        raise RuntimeError("Голос не виявлено")

    output_path = tmp_dir / "utterance.wav"
    merge_wavs(collected, str(output_path))
    return str(output_path)

def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не знайдено OPENAI_API_KEY у .env")

    client = OpenAI(api_key=api_key)

    print("[INIT] Ініціалізація Unitree channel...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    print("[INIT] Ініціалізація AudioClient...")
    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    audio_client.SetVolume(100)

    history: List[Dict] = []
    print("=== Unitree G1 Voice GPT ===")
    print("=== Voice auto mode ===")
    print("Ctrl+C = вийти")

    while True:
        try:
            wav_path = listen_until_silence_with_prebuffer(
                interface=NETWORK_INTERFACE,
                chunk_seconds=1.0,
                threshold=500,
                prebuffer_chunks=2,
                max_silence_chunks=2,
                max_total_chunks=15,
            )

            if not file_has_audio(wav_path):
                print("[INFO] Аудіо не записалось або файл порожній.")
                continue

            user_text = transcribe_audio(client, wav_path)

            if not user_text:
                print("[INFO] Нічого не розпізнано. Спробуй ще раз.")
                continue

            reply_text = ask_llm(client, history, user_text)

            time.sleep(0.3)
            speak_robot(audio_client, reply_text)
            time.sleep(1.5)

        except KeyboardInterrupt:
            print("\n[STOP] Зупинено користувачем.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
   


if __name__ == "__main__":
    main()
