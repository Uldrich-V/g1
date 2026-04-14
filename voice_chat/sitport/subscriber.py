from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def LowStateHandler(msg: LowState_):
    logger.info(f"RECEIVED on standard topic! Tick: {msg.tick}")

def LowStateHandlerLF(msg: LowState_):
    logger.info(f"RECEIVED on LF topic! Tick: {msg.tick}")

if __name__ == "__main__":
    interface = "en9"
    robot_ip = "192.168.123.164"
    domain_id = 0
    
    logger.info(f"Starting G1 diagnostic subscriber on {interface}...")

    try:
        # Inicializace s Peers a spdp (odpovídá konfiguraci G1)
        ChannelFactoryInitialize(domain_id, interface, peers=[robot_ip])
        
        # Monitorujeme obě témata, která G1 používá
        sub1 = ChannelSubscriber("rt/lowstate", LowState_)
        sub1.Init(LowStateHandler, 10)
        
        sub2 = ChannelSubscriber("rt/lf/lowstate", LowState_)
        sub2.Init(LowStateHandlerLF, 10)
        
        logger.info("Subscribers initialized. Waiting for messages (Ctrl+C to stop)...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Stopping...")
    except Exception as e:
        logger.exception(f"Error: {e}")