import struct
import time
import io

def resample_24_to_16(pcm_data):
    """
    Simple linear resampling from 24000Hz to 16000Hz.
    Ratio is 2:3.
    """
    # Convert bytes to list of 16-bit signed integers
    samples = []
    for i in range(0, len(pcm_data), 2):
        if i + 1 < len(pcm_data):
            sample = struct.unpack('<h', bytes(pcm_data[i:i+2]))[0]
            samples.append(sample)
    
    resampled = []
    # Every 3 samples at 24kHz become 2 samples at 16kHz
    for i in range(0, len(samples) // 3 * 3, 3):
        s0 = samples[i]
        s1 = samples[i+1]
        s2 = samples[i+2]
        
        # Linear interpolation
        # point 0 (at 0/3) -> s0
        # point 1 (at 1.5/3) -> (s1 + s2) / 2
        resampled.append(s0)
        resampled.append(int((s1 + s2) / 2))
        
    # Convert back to bytes
    out_bytes = []
    for s in resampled:
        out_bytes.extend(list(struct.pack('<h', s)))
    return out_bytes

def read_wav_from_bytes(wav_bytes):
    try:
        f = io.BytesIO(wav_bytes)
        def read(fmt):
            data = f.read(struct.calcsize(fmt))
            if not data:
                return None
            return struct.unpack(fmt, data)

        # === Chunk Header ===
        res = read('<I')
        if not res or res[0] != 0x46464952:  # "RIFF"
            return [], -1, -1, False

        read('<I') # chunk_size
        res = read('<I')
        if not res or res[0] != 0x45564157:  # "WAVE"
            return [], -1, -1, False

        # === Subchunk1: fmt ===
        while True:
            res = read('<I')
            if not res: return [], -1, -1, False
            subchunk_id = res[0]
            
            res = read('<I')
            if not res: return [], -1, -1, False
            subchunk_size = res[0]

            if subchunk_id == 0x20746D66:  # "fmt "
                break
            else:
                f.seek(subchunk_size, 1)

        audio_format, = read('<H')
        num_channels, = read('<H')
        sample_rate, = read('<I')
        byte_rate, = read('<I')
        block_align, = read('<H')
        bits_per_sample, = read('<H')

        if bits_per_sample != 16:
            return [], -1, -1, False

        if subchunk_size > 16:
            f.seek(subchunk_size - 16, 1)

        # === Subchunk2: data ===
        while True:
            res = read('<I')
            if not res: return [], -1, -1, False
            subchunk2_id = res[0]
            
            res = read('<I')
            if not res: return [], -1, -1, False
            subchunk2_size = res[0]

            if subchunk2_id == 0x61746164:  # "data"
                break
            f.seek(subchunk2_size, 1)

        # For streamed data, the subchunk2_size might be wrong or 0
        # We read until the end of the provided bytes
        raw_pcm = f.read()
        
        # If the rate is 24000, resample to 16000
        pcm_list = list(raw_pcm)
        if sample_rate == 24000:
            pcm_list = resample_24_to_16(pcm_list)
            sample_rate = 16000

        return pcm_list, sample_rate, num_channels, True

    except Exception as e:
        print(f"Error parsing WAV: {e}")
        return [], -1, -1, False

def read_wav(filename):
    try:
        with open(filename, 'rb') as f:
            return read_wav_from_bytes(f.read())
    except Exception:
        return [], -1, -1, False


def write_wave(filename, sample_rate, samples, num_channels=1):
    try:
        import array
        if isinstance(samples[0], int):
            samples = array.array('h', samples)

        subchunk2_size = len(samples) * 2
        chunk_size = 36 + subchunk2_size

        with open(filename, 'wb') as f:
            # RIFF chunk
            f.write(struct.pack('<I', 0x46464952))  # "RIFF"
            f.write(struct.pack('<I', chunk_size))
            f.write(struct.pack('<I', 0x45564157))  # "WAVE"

            # fmt subchunk
            f.write(struct.pack('<I', 0x20746D66))  # "fmt "
            f.write(struct.pack('<I', 16))          # PCM
            f.write(struct.pack('<H', 1))           # PCM format
            f.write(struct.pack('<H', num_channels))
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', sample_rate * num_channels * 2))  # byte_rate
            f.write(struct.pack('<H', num_channels * 2))                # block_align
            f.write(struct.pack('<H', 16))                              # bits per sample

            # data subchunk
            f.write(struct.pack('<I', 0x61746164))  # "data"
            f.write(struct.pack('<I', subchunk2_size))
            f.write(samples.tobytes())

        return True
    except Exception:
        return False


def play_pcm_stream(client, pcm_list, stream_name="example", chunk_size=96000, sleep_time=1.0, logger=None):
    """
    Play PCM audio stream (16-bit little-endian format), sending data in chunks.
    """
    pcm_data = bytes(pcm_list)
    stream_id = str(int(time.time() * 1000))  # Unique stream ID based on current timestamp
    offset = 0
    chunk_index = 0
    total_size = len(pcm_data)

    while offset < total_size:
        remaining = total_size - offset
        current_chunk_size = min(chunk_size, remaining)
        chunk = pcm_data[offset:offset + current_chunk_size]

        # Send the chunk
        ret_code, _ = client.PlayStream(stream_name, stream_id, chunk)
        if ret_code != 0:
            if logger:
                logger.error(f"Failed to send chunk {chunk_index}, return code: {ret_code}")
            break
        else:
            if logger:
                logger.info(f"Chunk {chunk_index} sent successfully")

        offset += current_chunk_size
        chunk_index += 1
        time.sleep(sleep_time)
