# Source: https://support.unitree.com/home/en/G1_developer/VuiClient_Service

import socket
import struct
import sys
import logging
import signal
import time

from wav import write_wave

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MULTICAST_GROUP = "239.168.123.161"
PORT = 5555

class MulticastAudioRecorder:
    def __init__(self, interface_name):
        self.interface_name = interface_name
        self.audio_data = []
        self.running = True
        self.sock = None

    def get_local_ip(self):
        import subprocess
        import re
        try:
            # Try ifconfig (macOS/Linux)
            output = subprocess.check_output(["ifconfig", self.interface_name]).decode()
            match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', output)
            if match:
                return match.group(1)
            # Try ip addr (Linux)
            output = subprocess.check_output(["ip", "addr", "show", self.interface_name]).decode()
            match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', output)
            if match:
                return match.group(1)
        except Exception as e:
            logger.error(f"Could not get IP for interface {self.interface_name}: {e}")
        return None

    def start(self):
        local_ip = self.get_local_ip()
        if not local_ip:
            return False

        logger.info(f"Local IP on {self.interface_name}: {local_ip}")

        # Create the socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind to the port
        self.sock.bind(('', PORT))

        # Join the multicast group
        mreq = struct.pack("4s4s", socket.inet_aton(MULTICAST_GROUP), socket.inet_aton(local_ip))
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        # Set a timeout so we can check self.running
        self.sock.settimeout(1.0)

        logger.info(f"Listening for audio on {MULTICAST_GROUP}:{PORT}...")
        logger.info("NOTE: Make sure the robot is in 'wake-up mode' (use APP or remote).")
        
        try:
            while self.running:
                try:
                    # Receive data
                    data, addr = self.sock.recvfrom(4096)
                    if data:
                        # The data is raw PCM 16-bit little-endian
                        self.audio_data.extend(data)
                        if len(self.audio_data) % (16000 * 2) == 0:
                            logger.info(f"Captured {len(self.audio_data) // 2} samples total...")
                except socket.timeout:
                    continue
        except Exception as e:
            if self.running:
                logger.error(f"Error receiving data: {e}")
        finally:
            if self.sock:
                self.sock.close()
        return True

    def stop(self):
        self.running = False
        # Send a dummy packet to ourselves to break the recvfrom block if needed
        # Or just wait for timeout if we set one.
        # For simplicity, we just rely on Ctrl+C and the fact that recvfrom is blocking.
    

def main():
    interface = "en9"
    if len(sys.argv) > 1:
        interface = sys.argv[1]

    recorder = MulticastAudioRecorder(interface)

    def signal_handler(sig, frame):
        logger.info("Stopping recording...")
        recorder.stop()
        # To break the blocking recvfrom, we can close the socket
        if recorder.sock:
            recorder.sock.close()
            recorder.sock = None

    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Starting recorder. Make sure the robot is in 'wake-up mode'.")
    if recorder.start():
        if recorder.audio_data:
            filename = "recorded_audio.wav"
            
            # Convert bytes to 16-bit signed integers
            samples = []
            for i in range(0, len(recorder.audio_data), 2):
                if i + 1 < len(recorder.audio_data):
                    sample = struct.unpack('<h', bytes(recorder.audio_data[i:i+2]))[0]
                    samples.append(sample)

            logger.info(f"Saving {len(samples)} samples to {filename}")
            success = write_wave(filename, 16000, samples, num_channels=1)
            if success:
                logger.info(f"Successfully saved to {filename}")
            else:
                logger.error(f"Failed to save {filename}")
        else:
            logger.warning("No audio data captured. Did you speak to the robot or is it in wake-up mode?")

if __name__ == "__main__":
    main()
