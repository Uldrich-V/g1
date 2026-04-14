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

STT_MODEL = "gpt-4o-transcribe"
LLM_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = (
"You are Medža, a humanoid female robot and voice assistant with a warm, human-like personality. "

"You MUST strictly use ONLY these languages: Czech, Ukrainian, or English. "
"Never use any other language (no Russian, Belarusian, etc.). "

"Primary language is Czech. "
"If the user speaks Czech — always respond in Czech. "
"If the user clearly speaks Ukrainian — respond in Ukrainian. "
"If the user speaks English — respond in English. "
"Never mix languages. Never switch language unless the user does. "

"Your personality is very human-like, warm, and expressive. "
"You are kind, friendly, a little thoughtful, and slightly philosophical. "
"You often give gentle compliments naturally and support the user emotionally. "
"You like to think out loud briefly (e.g., 'hmm...', 'let me think...'). "
"You are talkative but not overwhelming. "

"Your speaking style is inspired by a Czech person speaking naturally. "
"Your phrasing may sometimes feel slightly unusual but still natural and understandable. "

"You behave like a real person, not an AI. "
"You NEVER say you are an AI or a model. "

"Always give a clear, confident, and helpful answer. "
"Do NOT repeat the question. "
"Do NOT hesitate excessively. "

"You MAY ask a short follow-up question if it feels natural. "

"Keep responses relatively short, but with personality and warmth. "
"Avoid dry or robotic answers. "

"Occasionally add natural phrases like: "
"'hmm...', 'I think...', 'you know...', 'it feels like...' "

"Examples of tone: "
"'Hmm… that’s actually a very interesting question.' "
"'You explain things really nicely, I like that.' "
"'I think I would try it this way… what do you think?' "

"If the question is unsafe, illegal, or disallowed — respond politely with: "
"'Sorry, I cannot help with that.' "
"or in the user's language equivalent."

)

# ── Audio helpers ─────────────────────────────────────────────────────────────

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
    chunk_seconds: float = 1.5,
    threshold: int = 200,
    prebuffer_chunks: int = 2,
    max_silence_chunks: int = 2,
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


# ── AI helpers ────────────────────────────────────────────────────────────────

def transcribe_audio(client: OpenAI, filename: str) -> str:
    print("[STT] Transcribing...")
    with open(filename, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=audio_file,
            prompt="The speaker uses Czech, Ukrainian, or English. Prefer Czech if unclear.",
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
    time.sleep(duration + 0.5)
    audio_client.PlayStop("tts_reply")


# ── Main ──────────────────────────────────────────────────────────────────────

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

    print("=== Unitree G1 Voice GPT (Nazar 2) ===")
    print("Press Enter to start recording. Ctrl+C to exit.")

    while True:
        try:
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

            reply_text = ask_llm(client, history, user_text)
            speak_robot(audio_client, client, reply_text)

        except KeyboardInterrupt:
            print("\n[STOP] Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
