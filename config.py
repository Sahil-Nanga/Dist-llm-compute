"""
config.py — System-wide constants and configuration.
All ports, timeouts, and protocol identifiers live here.
"""

                                                                              
MASTER_API_PORT: int = 8000
MASTER_API_HOST: str = "0.0.0.0"

                                                                              
                                                    
ZMQ_CONTROL_PORT: int = 5555                                             
ZMQ_WEIGHT_PORT: int  = 5556                                             
ZMQ_PIPELINE_PORT: int = 5557                                                       

                                                                        
ZMQ_COLLECT_PORT: int = 5558                                    

                                                                              
ZEROCONF_SERVICE_TYPE: str  = "_llmdist._tcp.local."
ZEROCONF_DISCOVERY_TIMEOUT: float = 15.0                                             
ZEROCONF_HEARTBEAT_INTERVAL: float = 5.0                                     

                                                                              
PROTOCOL_VERSION: int = 2

                                                                              
CMD_HELLO            = b"HELLO"
CMD_RAM_REPORT       = b"RAM_REPORT"
CMD_ASSIGN_LAYERS    = b"ASSIGN_LAYERS"
CMD_WEIGHTS_READY    = b"WEIGHTS_READY"
CMD_SET_NEXT_HOP     = b"SET_NEXT_HOP"
CMD_START_INFERENCE  = b"START_INFERENCE"
CMD_INFERENCE_DONE   = b"INFERENCE_DONE"
CMD_RESET            = b"RESET"
CMD_ACK              = b"ACK"
CMD_ERROR            = b"ERROR"

                                                                               
DEFAULT_MAX_NEW_TOKENS: int   = 256
DEFAULT_TEMPERATURE: float    = 0.7
DEFAULT_TOP_P: float          = 0.9
DEFAULT_TOP_K: int            = 40

                                                                               
                                                                         
RAM_SAFETY_MARGIN: float = 0.75

RAM_OVERHEAD_BYTES: int = 2_048 * 1024 * 1024

                                               
MiB: int = 1024 * 1024

                                                     
ZMQ_MAX_WEIGHT_CHUNK: int = 256 * MiB

                                                                              
                                                               
ARCH_PREFIXES = ["llama", "falcon", "gpt2", "bloom", "mpt", "gptj", "gptneox", "starcoder", "phi"]