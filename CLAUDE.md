# VisualBase - Claude Session Context

> 최종 업데이트: 2026-01-30
> 상태: **Phase 8 완료** (151 tests)

## 프로젝트 역할

981파크 Portrait981 파이프라인의 **미디어 플랫폼**:
- 카메라/파일/RTSP 소스에서 프레임 스트리밍
- Ring Buffer로 메모리 효율적 버퍼링
- FFmpeg 기반 클립 추출
- IPC (FIFO, UDS, ZMQ) 통신

## 아키텍처

```
visualbase/
├── sources/       # FileSource, CameraSource, RTSPSource
├── core/          # Frame, RingBuffer, Sampler
├── packaging/     # Trigger, Clipper
├── ipc/           # FIFO, UDS, ZMQ (VideoReader/Writer, MessageSender/Receiver)
├── streaming/     # ProxyFanout
├── process/       # IngestProcess (A 모듈)
└── daemon.py      # ZMQ 데몬 모드
```

## 현재 구현 상태

### 완료
- Phase 1-7: 핵심 기능 (소스, 버퍼, 클립)
- Phase 8.0: IPC 인터페이스 추상화 (interfaces.py, factory.py)
- Phase 8.6: ZeroMQ Transport
- Phase 8.7: 데몬 모드
- Phase 8.8: WebRTC 출력
- Phase 8.9: GPU 가속 (nvdec/vaapi)

### 미완료
- Phase 8.4: IngestProcess 정리 (리팩토링 필요)
- Phase 8.5: CLI 확장 및 통합 테스트

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `src/visualbase/api.py` | VisualBase 메인 API |
| `src/visualbase/cli.py` | CLI (play, daemon, clip, etc.) |
| `src/visualbase/ipc/interfaces.py` | VideoReader/Writer ABC |
| `src/visualbase/ipc/zmq_transport.py` | ZMQ PUB/SUB |
| `src/visualbase/process/ingest.py` | IngestProcess (A 모듈) |
| `src/visualbase/daemon.py` | VideoDaemon |

## CLI 명령어

```bash
visualbase play <source>           # 통합 재생
visualbase daemon <source>         # ZMQ 데몬
visualbase clip <path> --time N    # 클립 추출
visualbase webrtc <source>         # 브라우저 스트리밍
visualbase ingest <source>         # A-B*-C 인제스트
```

## 테스트

```bash
cd ~/repo/monolith/visualbase
uv run pytest tests/ -v            # 151 tests
```

## 의존성

- 코어: opencv-python, numpy
- 옵션: pyzmq (zmq), aiortc (webrtc)

## 관련 패키지

- **facemoment**: visualbase를 import해서 사용 (Library 모드)
- **portrait981**: 프로덕션에서 orchestrator가 프로세스 관리

## 다음 작업 우선순위

1. IngestProcess 리팩토링 (인터페이스 의존)
2. CLI ingest 명령어 완성
3. facemoment와 통합 테스트
