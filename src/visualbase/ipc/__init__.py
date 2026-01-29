"""IPC (Inter-Process Communication) module for A-B*-C architecture.

This module provides Unix-native IPC primitives:
- FIFO: Named pipes for video stream fan-out (A→B*)
- UDS: Unix Domain Sockets for message passing (B*→C, C→A)
- Messages: OBS/TRIG message parsing and serialization
- Interfaces: ABCs for swappable transport implementations
- Factory: Unified transport creation

Example using interfaces (recommended for production code):
    >>> from visualbase.ipc import VideoReader, TransportFactory
    >>>
    >>> def process_frames(reader: VideoReader):
    ...     reader.open()
    ...     for frame in reader:
    ...         process(frame)
    ...     reader.close()
    >>>
    >>> reader = TransportFactory.create_video_reader("fifo", "/tmp/vid.mjpg")
    >>> process_frames(reader)

Example using concrete classes (fine for direct usage):
    >>> from visualbase.ipc import FIFOVideoReader, UDSClient
    >>> reader = FIFOVideoReader("/tmp/vid.mjpg")
    >>> client = UDSClient("/tmp/obs.sock")
"""

# Interfaces (ABCs)
from visualbase.ipc.interfaces import (
    VideoReader,
    VideoWriter,
    MessageReceiver,
    MessageSender,
)

# Concrete implementations
from visualbase.ipc.fifo import FIFOVideoWriter, FIFOVideoReader
from visualbase.ipc.uds import UDSServer, UDSClient

# Factory
from visualbase.ipc.factory import TransportFactory

# Messages
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
    # Interfaces
    "VideoReader",
    "VideoWriter",
    "MessageReceiver",
    "MessageSender",
    # Factory
    "TransportFactory",
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
