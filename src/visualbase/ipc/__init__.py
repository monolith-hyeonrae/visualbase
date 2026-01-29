"""IPC (Inter-Process Communication) module for A-B*-C architecture.

This module provides Unix-native IPC primitives:
- FIFO: Named pipes for video stream fan-out (A→B*)
- UDS: Unix Domain Sockets for message passing (B*→C, C→A)
- Messages: OBS/TRIG message parsing and serialization
"""

from visualbase.ipc.fifo import FIFOVideoWriter, FIFOVideoReader
from visualbase.ipc.uds import UDSServer, UDSClient
from visualbase.ipc.messages import (
    OBSMessage,
    TRIGMessage,
    parse_obs_message,
    parse_trig_message,
    FaceOBS,
    PoseOBS,
    QualityOBS,
)

__all__ = [
    # FIFO
    "FIFOVideoWriter",
    "FIFOVideoReader",
    # UDS
    "UDSServer",
    "UDSClient",
    # Messages
    "OBSMessage",
    "TRIGMessage",
    "parse_obs_message",
    "parse_trig_message",
    "FaceOBS",
    "PoseOBS",
    "QualityOBS",
]
