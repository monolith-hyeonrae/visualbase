"""Command-line interface for visualbase."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="VisualBase - Video frame streaming and visualization"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # play command
    play_parser = subparsers.add_parser("play", help="Play a video file")
    play_parser.add_argument("path", help="Path to video file")
    play_parser.add_argument(
        "--fps", type=int, default=0, help="Target FPS (0 for original)"
    )
    play_parser.add_argument(
        "--width", type=int, default=0, help="Target width (0 for original)"
    )
    play_parser.add_argument(
        "--height", type=int, default=0, help="Target height (0 for original)"
    )
    play_parser.add_argument(
        "--no-info", action="store_true", help="Hide frame info overlay"
    )

    # info command
    info_parser = subparsers.add_parser("info", help="Show video file info")
    info_parser.add_argument("path", help="Path to video file")

    # camera command
    camera_parser = subparsers.add_parser("camera", help="Test USB camera connection")
    camera_parser.add_argument(
        "device", nargs="?", type=int, default=0,
        help="Camera device ID (default: 0 for /dev/video0)"
    )
    camera_parser.add_argument(
        "--fps", type=int, default=30, help="Target FPS (default: 30)"
    )
    camera_parser.add_argument(
        "--width", type=int, default=0, help="Target width (0 for camera default)"
    )
    camera_parser.add_argument(
        "--height", type=int, default=0, help="Target height (0 for camera default)"
    )
    camera_parser.add_argument(
        "--buffer", type=float, default=60.0,
        help="Ring buffer retention in seconds (default: 60)"
    )
    camera_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips",
        help="Output directory for clips (default: ./clips)"
    )
    camera_parser.add_argument(
        "--no-window", action="store_true",
        help="Headless mode (no GUI window)"
    )

    # rtsp command
    rtsp_parser = subparsers.add_parser("rtsp", help="Test RTSP stream connection")
    rtsp_parser.add_argument("url", help="RTSP stream URL")
    rtsp_parser.add_argument(
        "--fps", type=int, default=30, help="Target FPS (default: 30)"
    )
    rtsp_parser.add_argument(
        "--buffer", type=float, default=60.0,
        help="Ring buffer retention in seconds (default: 60)"
    )
    rtsp_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips",
        help="Output directory for clips (default: ./clips)"
    )
    rtsp_parser.add_argument(
        "--no-reconnect", action="store_true",
        help="Disable auto-reconnection on stream loss"
    )
    rtsp_parser.add_argument(
        "--no-window", action="store_true",
        help="Headless mode (no GUI window)"
    )

    # clip command
    clip_parser = subparsers.add_parser("clip", help="Extract a clip from video")
    clip_parser.add_argument("path", help="Path to video file")
    clip_parser.add_argument(
        "--time", "-t", type=float, required=True, help="Event time in seconds"
    )
    clip_parser.add_argument(
        "--pre", type=float, default=3.0, help="Seconds before event (default: 3.0)"
    )
    clip_parser.add_argument(
        "--post", type=float, default=2.0, help="Seconds after event (default: 2.0)"
    )
    clip_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips", help="Output directory"
    )
    clip_parser.add_argument(
        "--label", "-l", type=str, default="", help="Label for the clip"
    )

    # ingest command (Phase 8: A-B*-C architecture)
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run ingest process (A module) for A-B*-C architecture"
    )
    ingest_parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera device ID (e.g., 0 for /dev/video0)"
    )
    ingest_parser.add_argument(
        "--rtsp", type=str, default=None,
        help="RTSP stream URL"
    )
    ingest_parser.add_argument(
        "--trig-socket", type=str, default="/tmp/trig.sock",
        help="UDS socket path for receiving TRIG messages"
    )
    ingest_parser.add_argument(
        "--proxy", action="append", default=[],
        help="Proxy output config: name:path:width:height:fps (can specify multiple)"
    )
    ingest_parser.add_argument(
        "--buffer", type=float, default=120.0,
        help="Ring buffer retention in seconds (default: 120)"
    )
    ingest_parser.add_argument(
        "--output-dir", "-o", type=str, default="./clips",
        help="Output directory for clips (default: ./clips)"
    )

    args = parser.parse_args()

    if args.command == "play":
        from visualbase.tools.viewer import play

        resolution = None
        if args.width > 0 and args.height > 0:
            resolution = (args.width, args.height)
        elif args.width > 0 or args.height > 0:
            print("Error: Both --width and --height must be specified together")
            sys.exit(1)

        play(
            args.path,
            fps=args.fps,
            resolution=resolution,
            show_info=not args.no_info,
        )

    elif args.command == "info":
        from visualbase import FileSource

        try:
            with FileSource(args.path) as source:
                print(f"File: {source.path}")
                print(f"FPS: {source.fps:.2f}")
                print(f"Resolution: {source.width}x{source.height}")
                print(f"Frames: {source.frame_count}")
                print(f"Duration: {source.duration_sec:.2f}s")
                print(f"Seekable: {source.is_seekable}")
        except IOError as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "clip":
        from pathlib import Path
        from visualbase import VisualBase, FileSource, Trigger

        try:
            event_time_ns = int(args.time * 1_000_000_000)

            with VisualBase(clip_output_dir=args.output_dir) as vb:
                vb.connect(FileSource(args.path))

                trig = Trigger.point(
                    event_time_ns=event_time_ns,
                    pre_sec=args.pre,
                    post_sec=args.post,
                    label=args.label or "clip",
                )

                print(f"Extracting clip:")
                print(f"  Source: {args.path}")
                print(f"  Event time: {args.time:.2f}s")
                print(f"  Clip range: [{trig.clip_start_sec:.2f}s, {trig.clip_end_sec:.2f}s]")
                print(f"  Duration: {trig.clip_duration_sec:.2f}s")

                result = vb.trigger(trig)

                if result.success:
                    print(f"  Output: {result.output_path}")
                    print(f"  Actual duration: {result.duration_sec:.2f}s")
                else:
                    print(f"  Error: {result.error}")
                    sys.exit(1)

        except IOError as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "camera":
        _run_camera(args)

    elif args.command == "rtsp":
        _run_rtsp(args)

    elif args.command == "ingest":
        _run_ingest(args)

    else:
        parser.print_help()
        sys.exit(1)


def _run_camera(args):
    """Run camera debug/test mode."""
    import cv2
    from pathlib import Path
    from visualbase import VisualBase, CameraSource, Trigger

    # Camera settings
    width = args.width if args.width > 0 else None
    height = args.height if args.height > 0 else None

    print(f"Connecting to camera device {args.device} (/dev/video{args.device})...", flush=True)

    try:
        source = CameraSource(
            device_id=args.device,
            width=width,
            height=height,
            fps=args.fps,
        )
    except Exception as e:
        print(f"Error creating camera source: {e}")
        sys.exit(1)

    vb = VisualBase(clip_output_dir=args.output_dir)

    try:
        vb.connect(source, ring_buffer_retention_sec=args.buffer)
    except IOError as e:
        print(f"Error: Failed to open camera: {e}")
        sys.exit(1)

    print(f"Camera connected:", flush=True)
    print(f"  Device: /dev/video{args.device}", flush=True)
    print(f"  Resolution: {vb.source.width}x{vb.source.height}", flush=True)
    print(f"  FPS: {vb.source.fps:.1f}", flush=True)
    print(f"  Buffer: {args.buffer}s", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print("-" * 50, flush=True)

    if args.no_window:
        print("Headless mode - press Ctrl+C to stop", flush=True)
        _run_headless(vb, args)
    else:
        print("Controls:", flush=True)
        print("  q: Quit", flush=True)
        print("  s: Save clip (3s before, 1s after)", flush=True)
        print("  i: Show buffer info", flush=True)
        print("-" * 50, flush=True)
        _run_with_window(vb, args, "Camera")


def _run_rtsp(args):
    """Run RTSP stream debug/test mode."""
    import cv2
    from pathlib import Path
    from visualbase import VisualBase, RTSPSource, Trigger

    print(f"Connecting to RTSP stream: {args.url}", flush=True)

    try:
        source = RTSPSource(
            url=args.url,
            reconnect=not args.no_reconnect,
        )
    except Exception as e:
        print(f"Error creating RTSP source: {e}")
        sys.exit(1)

    vb = VisualBase(clip_output_dir=args.output_dir)

    try:
        vb.connect(source, ring_buffer_retention_sec=args.buffer)
    except IOError as e:
        print(f"Error: Failed to connect to stream: {e}")
        sys.exit(1)

    print(f"Stream connected:", flush=True)
    print(f"  URL: {args.url}", flush=True)
    print(f"  Resolution: {vb.source.width}x{vb.source.height}", flush=True)
    print(f"  FPS: {vb.source.fps:.1f}", flush=True)
    print(f"  Buffer: {args.buffer}s", flush=True)
    print(f"  Reconnect: {not args.no_reconnect}", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print("-" * 50, flush=True)

    if args.no_window:
        print("Headless mode - press Ctrl+C to stop", flush=True)
        _run_headless(vb, args)
    else:
        print("Controls:", flush=True)
        print("  q: Quit", flush=True)
        print("  s: Save clip (3s before, 1s after)", flush=True)
        print("  i: Show buffer info", flush=True)
        print("-" * 50, flush=True)
        _run_with_window(vb, args, "RTSP")


def _run_with_window(vb, args, source_type: str):
    """Run with GUI window for visual feedback."""
    import cv2
    from visualbase import Trigger

    frame_count = 0
    clip_count = 0
    window_name = f"{source_type} Debug - q:quit s:save i:info"

    try:
        for frame in vb.get_stream(fps=args.fps):
            frame_count += 1

            # Overlay info on frame
            display = frame.data.copy()
            info = vb.get_buffer_info()

            # Status line
            status = f"Frame: {frame_count} | Buffer: {info.duration_sec:.1f}s | Clips: {clip_count}"
            cv2.putText(display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Time line
            time_str = f"Time: {frame.t_src_ns / 1e9:.2f}s"
            cv2.putText(display, time_str, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show frame
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuitting...")
                break

            elif key == ord("s"):
                # Save clip
                print(f"\nSaving clip at t={frame.t_src_ns / 1e9:.2f}s...")
                trigger = Trigger.point(
                    event_time_ns=frame.t_src_ns,
                    pre_sec=3.0,
                    post_sec=1.0,
                    label="manual",
                )
                result = vb.trigger(trigger)
                if result.success:
                    clip_count += 1
                    print(f"  Saved: {result.output_path} ({result.duration_sec:.2f}s)")
                else:
                    print(f"  Failed: {result.error}")

            elif key == ord("i"):
                # Show buffer info
                info = vb.get_buffer_info()
                print(f"\nBuffer Info:")
                print(f"  Range: [{info.start_ns / 1e9:.2f}s, {info.end_ns / 1e9:.2f}s]")
                print(f"  Duration: {info.duration_sec:.2f}s")
                print(f"  Seekable: {info.is_seekable}")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        cv2.destroyAllWindows()
        vb.disconnect()
        print(f"\nTotal frames: {frame_count}")
        print(f"Total clips: {clip_count}")


def _run_headless(vb, args):
    """Run in headless mode without GUI."""
    from visualbase import Trigger

    frame_count = 0
    last_report = 0

    try:
        for frame in vb.get_stream(fps=args.fps):
            frame_count += 1

            # Report every 30 frames
            if frame_count - last_report >= 30:
                info = vb.get_buffer_info()
                print(f"Frame {frame_count}: t={frame.t_src_ns / 1e9:.2f}s, buffer={info.duration_sec:.1f}s", flush=True)
                last_report = frame_count

    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)

    finally:
        vb.disconnect()
        print(f"\nTotal frames: {frame_count}", flush=True)


def _run_ingest(args):
    """Run ingest process (A module) for A-B*-C architecture."""
    from pathlib import Path
    from visualbase.sources.camera import CameraSource
    from visualbase.sources.rtsp import RTSPSource
    from visualbase.streaming.fanout import ProxyConfig
    from visualbase.process.ingest import IngestProcess

    # Determine source
    if args.camera is not None:
        print(f"Using camera device {args.camera}...", flush=True)
        source = CameraSource(device_id=args.camera)
    elif args.rtsp is not None:
        print(f"Using RTSP stream: {args.rtsp}", flush=True)
        source = RTSPSource(url=args.rtsp)
    else:
        print("Error: Must specify --camera or --rtsp")
        sys.exit(1)

    # Parse proxy configs
    proxy_configs = []
    for proxy_str in args.proxy:
        try:
            parts = proxy_str.split(":")
            if len(parts) != 5:
                raise ValueError(f"Invalid format: {proxy_str}")
            name, path, width, height, fps = parts
            proxy_configs.append(ProxyConfig(
                name=name,
                fifo_path=path,
                width=int(width),
                height=int(height),
                fps=float(fps),
            ))
        except ValueError as e:
            print(f"Error parsing proxy config '{proxy_str}': {e}")
            print("Expected format: name:path:width:height:fps")
            sys.exit(1)

    # Default proxy configs if none specified
    if not proxy_configs:
        proxy_configs = [
            ProxyConfig("face", "/tmp/vid_face.mjpg", 640, 480, 10),
            ProxyConfig("pose", "/tmp/vid_pose.mjpg", 640, 480, 10),
            ProxyConfig("quality", "/tmp/vid_quality.mjpg", 320, 240, 5),
        ]
        print("Using default proxy configs:", flush=True)
        for cfg in proxy_configs:
            print(f"  {cfg.name}: {cfg.fifo_path} ({cfg.width}x{cfg.height}@{cfg.fps}fps)")

    print(f"Ingest process configuration:", flush=True)
    print(f"  TRIG socket: {args.trig_socket}", flush=True)
    print(f"  Buffer: {args.buffer}s", flush=True)
    print(f"  Output: {args.output_dir}", flush=True)
    print("-" * 50, flush=True)

    # Create and run ingest process
    process = IngestProcess(
        source=source,
        proxy_configs=proxy_configs,
        trig_socket=args.trig_socket,
        clip_output_dir=Path(args.output_dir),
        ring_buffer_retention_sec=args.buffer,
    )

    try:
        process.run()
    except KeyboardInterrupt:
        print("\nInterrupted", flush=True)
    finally:
        stats = process.get_stats()
        print(f"\nIngest process stats:", flush=True)
        print(f"  Frames: {stats['frames_captured']}", flush=True)
        print(f"  FPS: {stats['fps']:.1f}", flush=True)
        print(f"  Triggers: {stats['triggers_received']}", flush=True)
        print(f"  Clips: {stats['clips_extracted']}", flush=True)
        print(f"  Errors: {stats['errors']}", flush=True)


if __name__ == "__main__":
    main()
