import asyncio
import json
import logging
from pathlib import Path
import numpy as np
import av

import uvicorn
from aiortc import (
    RTCPeerConnection, RTCSessionDescription, RTCIceCandidate,
    RTCConfiguration, RTCIceServer
)
from aiortc.contrib.media import MediaStreamTrack
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright
from PIL import Image
from fractions import Fraction
import io

ROOT = Path(__file__).parent
VIEWPORT = {"width": 1280, "height": 720}
pcs = set()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BrowserFrameTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self):
        super().__init__()
        self.page = None
        self.browser = None
        self.playwright = None
        self.playwright_task = None
        self._stop_event = asyncio.Event()
        self.frame_rate = 10
        self.frame_time = 1.0 / self.frame_rate

    async def start_playwright(self):
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=True)
            self.page = await self.browser.new_page(viewport=VIEWPORT)
            await self.page.goto("https://www.google.com", wait_until="domcontentloaded")
            logging.info("Playwright browser launched.")
        except Exception as e:
            logging.error(f"Failed to start Playwright: {e}")
            await self.stop()

    async def recv(self):
        if self._stop_event.is_set() or not self.page:
            return av.VideoFrame(width=VIEWPORT["width"], height=VIEWPORT["height"], format="rgb24")
        try:
            screenshot_bytes = await self.page.screenshot(type="png")
        except Exception as e:
            logging.error(f"Failed to capture screenshot: {e}")
            await self.stop()
            return av.VideoFrame(width=VIEWPORT["width"], height=VIEWPORT["height"], format="rgb24")
        img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        frame = self.create_frame_from_image(img)
        await asyncio.sleep(self.frame_time)
        return frame

    def create_frame_from_image(self, img):
        img_array = np.array(img)
        frame = av.VideoFrame.from_ndarray(img_array, format='rgb24')
        frame.pts = self._next_timestamp()
        frame.time_base = Fraction(1, 1000000)
        return frame

    def _next_timestamp(self):
        if not hasattr(self, '_timestamp'):
            self._timestamp = 0
        else:
            self._timestamp += int(self.frame_time * 1000000)
        return self._timestamp

    async def stop(self):
        if not self._stop_event.is_set():
            super().stop()
            self._stop_event.set()
            if self.playwright_task and not self.playwright_task.done():
                self.playwright_task.cancel()
            try:
                if self.page and not self.page.is_closed():
                    await self.page.close()
                if self.browser:
                    await self.browser.close()
                if self.playwright:
                    await self.playwright.stop()
                logging.info("Playwright resources cleaned up.")
            except Exception as e:
                logging.error(f"Error during playwright cleanup: {e}")

def parse_ice_candidate(candidate_string: str):
    try:
        if candidate_string.startswith('candidate:'):
            candidate_string = candidate_string[10:]
        parts = candidate_string.split()
        if len(parts) < 8:
            logging.warning(f"Could not parse candidate, too few parts: {candidate_string}")
            return None
        return {
            "foundation": parts[0],
            "component": int(parts[1]),
            "protocol": parts[2].lower(),
            "priority": int(parts[3]),
            "ip": parts[4],
            "port": int(parts[5]),
            "type": parts[7],
        }
    except (IndexError, ValueError) as e:
        logging.error(f"Failed to parse ICE candidate string '{candidate_string}': {e}")
        return None

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Client connected for signaling.")

    ice_servers = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(urls=["turn:your.turn.server:3478"], username="user", credential="pass")
]
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
    video_track = BrowserFrameTrack()
    pc.addTrack(video_track)
    pcs.add(pc)
    video_track.playwright_task = asyncio.create_task(video_track.start_playwright())

    async def cleanup():
        logging.info("Cleaning up connection.")
        try:
            await video_track.stop()
            if pc.connectionState != "closed":
                await pc.close()
            if pc in pcs:
                pcs.remove(pc)
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logging.info(f"Connection state is {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await cleanup()

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logging.info(f"ICE connection state is {pc.iceConnectionState}")

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type")
            if msg_type == "offer":
                logging.info("Received WebRTC offer from client.")
                offer = RTCSessionDescription(sdp=data["sdp"]["sdp"], type=data["sdp"]["type"])
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await websocket.send_json({
                    "type": "answer",
                    "sdp": {
                        "sdp": pc.localDescription.sdp,
                        "type": pc.localDescription.type
                    }
                })
                logging.info("Sent WebRTC answer to client.")

            elif msg_type == "ice-candidate" and data.get("candidate"):
                candidate_info = data.get("candidate")
                candidate_string = candidate_info.get("candidate")
                if candidate_string:
                    parsed_candidate = parse_ice_candidate(candidate_string)
                    if parsed_candidate:
                        try:
                            ice_candidate = RTCIceCandidate(
                                sdpMid=candidate_info.get("sdpMid"),
                                sdpMLineIndex=candidate_info.get("sdpMLineIndex"),
                                **parsed_candidate
                            )
                            await pc.addIceCandidate(ice_candidate)
                            logging.info("Added ICE candidate from client.")
                        except Exception as e:
                            logging.error(f"Error adding parsed ICE candidate: {e}")
                else:
                    logging.warning("Received empty ICE candidate from client.")

            elif video_track.page and not video_track.page.is_closed():
                if msg_type == "click":
                    await video_track.page.mouse.click(data["x"], data["y"])
                elif msg_type == "type":
                    await video_track.page.keyboard.type(data["key"], delay=10)
                elif msg_type == "scroll":
                    await video_track.page.mouse.wheel(delta_x=data["deltaX"], delta_y=data["deltaY"])
                elif msg_type == "navigate":
                    url_to_go = data.get("url", "https://google.com")
                    if not url_to_go.startswith(('http://', 'https://')):
                        url_to_go = 'https://' + url_to_go
                    logging.info(f"Navigating to: {url_to_go}")
                    await video_track.page.goto(url_to_go, wait_until="networkidle")

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        await cleanup()

@app.on_event("shutdown")
async def on_shutdown():
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("aioice").setLevel(logging.WARNING)
    uvicorn.run(app, host="127.0.0.1", port=3001)
