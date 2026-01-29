"""Tests for IPC modules (FIFO, UDS, messages)."""

import os
import tempfile
import threading
import time
from pathlib import Path

import pytest
import numpy as np

from visualbase.core.frame import Frame
from visualbase.ipc.fifo import FIFOVideoWriter, FIFOVideoReader, HEADER_SIZE
from visualbase.ipc.uds import UDSServer, UDSClient
from visualbase.ipc.messages import (
    FaceData,
    PoseData,
    QualityData,
    FaceOBS,
    PoseOBS,
    QualityOBS,
    OBSMessage,
    TRIGMessage,
    parse_obs_message,
    parse_trig_message,
)


class TestFaceData:
    """Tests for FaceData serialization."""

    def test_to_string(self):
        face = FaceData(
            id=0, conf=0.95, x=0.1, y=0.2, w=0.3, h=0.4, expr=0.8
        )
        s = face.to_string()
        assert "id:0" in s
        assert "conf:0.950" in s
        assert "expr:0.800" in s

    def test_from_string(self):
        s = "id:0,conf:0.95,x:0.1,y:0.2,w:0.3,h:0.4,expr:0.8"
        face = FaceData.from_string(s)
        assert face.id == 0
        assert face.conf == 0.95
        assert face.x == 0.1
        assert face.expr == 0.8

    def test_roundtrip(self):
        original = FaceData(
            id=1, conf=0.92, x=0.15, y=0.25, w=0.35, h=0.45,
            expr=0.7, yaw=5.0, pitch=2.0
        )
        s = original.to_string()
        parsed = FaceData.from_string(s)
        assert parsed.id == original.id
        assert parsed.conf == pytest.approx(original.conf, rel=1e-2)
        assert parsed.expr == pytest.approx(original.expr, rel=1e-2)


class TestPoseData:
    """Tests for PoseData serialization."""

    def test_to_string(self):
        pose = PoseData(id=0, conf=0.9, hand_raised=True, hand_wave=False)
        s = pose.to_string()
        assert "id:0" in s
        assert "hr:1" in s
        assert "hw:0" in s

    def test_from_string(self):
        s = "id:0,conf:0.9,hr:1,hw:0"
        pose = PoseData.from_string(s)
        assert pose.id == 0
        assert pose.hand_raised is True
        assert pose.hand_wave is False


class TestQualityData:
    """Tests for QualityData serialization."""

    def test_to_string(self):
        quality = QualityData(blur=100.0, brightness=128.0, contrast=0.5, gate_open=True)
        s = quality.to_string()
        assert "blur:100.0" in s
        assert "bright:128.0" in s
        assert "gate:1" in s

    def test_from_string(self):
        s = "blur:100.0,bright:128.0,contrast:0.500,gate:1"
        quality = QualityData.from_string(s)
        assert quality.blur == 100.0
        assert quality.brightness == 128.0
        assert quality.gate_open is True


class TestFaceOBS:
    """Tests for FaceOBS message."""

    def test_to_message(self):
        obs = FaceOBS(
            frame_id=1234,
            t_ns=1234567890,
            faces=[
                FaceData(id=0, conf=0.95, x=0.1, y=0.2, w=0.3, h=0.4, expr=0.8),
            ],
        )
        msg = obs.to_message()
        assert msg.startswith("OBS src=face")
        assert "frame=1234" in msg
        assert "faces=1" in msg
        assert "f0=" in msg


class TestPoseOBS:
    """Tests for PoseOBS message."""

    def test_to_message(self):
        obs = PoseOBS(
            frame_id=1234,
            t_ns=1234567890,
            poses=[
                PoseData(id=0, conf=0.9, hand_raised=True, hand_wave=False),
            ],
        )
        msg = obs.to_message()
        assert msg.startswith("OBS src=pose")
        assert "frame=1234" in msg
        assert "poses=1" in msg
        assert "p0=" in msg


class TestQualityOBS:
    """Tests for QualityOBS message."""

    def test_to_message(self):
        obs = QualityOBS(
            frame_id=1234,
            t_ns=1234567890,
            quality=QualityData(blur=100.0, brightness=128.0, contrast=0.5),
        )
        msg = obs.to_message()
        assert msg.startswith("OBS src=quality")
        assert "frame=1234" in msg
        assert "blur:100.0" in msg


class TestParseOBSMessage:
    """Tests for parse_obs_message."""

    def test_parse_face_obs(self):
        msg = "OBS src=face frame=1234 t_ns=1234567890 faces=1 f0=id:0,conf:0.95,x:0.1,y:0.2,w:0.3,h:0.4,expr:0.8"
        obs = parse_obs_message(msg)
        assert obs is not None
        assert obs.src == "face"
        assert obs.frame_id == 1234
        assert len(obs.faces) == 1
        assert obs.faces[0].id == 0
        assert obs.faces[0].expr == 0.8

    def test_parse_pose_obs(self):
        msg = "OBS src=pose frame=1234 t_ns=1234567890 poses=1 p0=id:0,conf:0.9,hr:1,hw:0"
        obs = parse_obs_message(msg)
        assert obs is not None
        assert obs.src == "pose"
        assert len(obs.poses) == 1
        assert obs.poses[0].hand_raised is True

    def test_parse_invalid(self):
        assert parse_obs_message("INVALID") is None
        assert parse_obs_message("") is None


class TestTRIGMessage:
    """Tests for TRIGMessage."""

    def test_to_message(self):
        trig = TRIGMessage(
            label="PORTRAIT_HIGHLIGHT",
            t_start_ns=1234567890,
            t_end_ns=1239567890,
            faces=2,
            score=0.85,
            reason="expr_spike",
        )
        msg = trig.to_message()
        assert msg.startswith("TRIG")
        assert "label=PORTRAIT_HIGHLIGHT" in msg
        assert "score=0.850" in msg
        assert "reason=expr_spike" in msg


class TestParseTRIGMessage:
    """Tests for parse_trig_message."""

    def test_parse_trig(self):
        msg = "TRIG label=PORTRAIT_HIGHLIGHT t_start_ns=1234567890 t_end_ns=1239567890 faces=2 score=0.85 reason=expr_spike"
        trig = parse_trig_message(msg)
        assert trig is not None
        assert trig.label == "PORTRAIT_HIGHLIGHT"
        assert trig.t_start_ns == 1234567890
        assert trig.t_end_ns == 1239567890
        assert trig.faces == 2
        assert trig.score == 0.85
        assert trig.reason == "expr_spike"

    def test_parse_invalid(self):
        assert parse_trig_message("INVALID") is None


class TestUDS:
    """Tests for UDS server and client."""

    def test_server_start_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")
            server = UDSServer(sock_path)
            server.start()
            assert server.is_running
            assert Path(sock_path).exists()
            server.stop()
            assert not server.is_running

    def test_client_connect_disconnect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")
            server = UDSServer(sock_path)
            server.start()

            client = UDSClient(sock_path)
            assert client.connect()
            assert client.is_connected
            client.disconnect()
            assert not client.is_connected

            server.stop()

    def test_send_receive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")

            with UDSServer(sock_path) as server:
                with UDSClient(sock_path) as client:
                    # Send message
                    assert client.send("Hello, World!")

                    # Receive message
                    msg = server.recv(timeout=1.0)
                    assert msg == "Hello, World!"

    def test_recv_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sock_path = os.path.join(tmpdir, "test.sock")

            with UDSServer(sock_path) as server:
                with UDSClient(sock_path) as client:
                    # Send multiple messages
                    client.send("msg1")
                    client.send("msg2")
                    client.send("msg3")

                    # Give messages time to arrive
                    time.sleep(0.1)

                    # Receive all
                    messages = server.recv_all()
                    assert len(messages) == 3
                    assert "msg1" in messages
                    assert "msg2" in messages
                    assert "msg3" in messages


class TestFIFO:
    """Tests for FIFO video streaming."""

    def _create_test_frame(self, frame_id: int = 1) -> Frame:
        """Create a test frame."""
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        data[100:200, 100:200] = (255, 0, 0)  # Red square
        return Frame.from_array(data, frame_id=frame_id, t_src_ns=frame_id * 1_000_000_000)

    def test_writer_reader_roundtrip(self):
        """Test writing and reading frames through FIFO."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo_path = os.path.join(tmpdir, "test.mjpg")

            # Start writer in background (blocks until reader connects)
            writer = FIFOVideoWriter(fifo_path)
            writer_thread = threading.Thread(target=writer.open)
            writer_thread.start()

            # Give writer time to create FIFO
            time.sleep(0.1)

            # Open reader (unblocks writer)
            reader = FIFOVideoReader(fifo_path)
            assert reader.open()

            # Wait for writer to finish connecting
            writer_thread.join(timeout=1.0)

            # Write a frame
            original = self._create_test_frame(1)
            assert writer.write(original)

            # Read the frame
            received = reader.read()
            assert received is not None
            assert received.frame_id == 1
            assert received.t_src_ns == 1_000_000_000
            assert received.width == 640
            assert received.height == 480

            # Clean up
            writer.close()
            reader.close()

    def test_multiple_frames(self):
        """Test writing and reading multiple frames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fifo_path = os.path.join(tmpdir, "test.mjpg")

            writer = FIFOVideoWriter(fifo_path)
            writer_thread = threading.Thread(target=writer.open)
            writer_thread.start()

            time.sleep(0.1)

            reader = FIFOVideoReader(fifo_path)
            reader.open()
            writer_thread.join(timeout=1.0)

            # Write multiple frames
            for i in range(5):
                frame = self._create_test_frame(i)
                writer.write(frame)

            # Read them back
            for i in range(5):
                received = reader.read()
                assert received is not None
                assert received.frame_id == i

            writer.close()
            reader.close()
