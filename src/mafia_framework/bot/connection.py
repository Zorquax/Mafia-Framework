import json
import logging
import urllib.parse
import requests
import websockets
import asyncio

logger = logging.getLogger("mafia_bot.connection")

class ShowdownConnection:
    def __init__(self, server_url: str, login_url: str, username: str, password: str, room: str):
        self.server_url = server_url
        self.login_url = login_url
        self.username = username
        self.password = password
        self.room = room
        self.websocket = None
        self.receive_queue = asyncio.Queue()
        self.send_queue = asyncio.Queue()
        self._running = False
        self._connected = False

    async def connect(self):
        self._running = True
        logger.info(f"Connecting to Showdown server at {self.server_url}...")
        while self._running:
            try:
                async with websockets.connect(self.server_url) as ws:
                    self.websocket = ws
                    self._connected = True
                    logger.info("Connected to Showdown websocket.")
                    
                    # Spawn send and receive tasks
                    send_task = asyncio.create_task(self._send_loop())
                    receive_task = asyncio.create_task(self._receive_loop())
                    
                    # Wait for either to finish (or error out)
                    done, pending = await asyncio.wait(
                        [send_task, receive_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Clean up pending tasks
                    for task in pending:
                        task.cancel()
                    
            except Exception as e:
                logger.error(f"Websocket connection error: {e}")
            
            self._connected = False
            self.websocket = None
            if self._running:
                logger.info("Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def disconnect(self):
        self._running = False
        if self.websocket:
            await self.websocket.close()

    async def send(self, message: str):
        await self.send_queue.put(message)

    async def _send_loop(self):
        while self._running and self._connected:
            try:
                msg = await self.send_queue.get()
                if self.websocket:
                    logger.debug(f"Sending: {msg}")
                    await self.websocket.send(msg)
                self.send_queue.task_done()
            except Exception as e:
                logger.error(f"Error in send loop: {e}")
                break

    async def _receive_loop(self):
        while self._running and self._connected:
            try:
                raw_msg = await self.websocket.recv()
                logger.debug(f"Received raw: {raw_msg}")
                await self._handle_raw_message(raw_msg)
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                break

    async def _handle_raw_message(self, raw_msg: str):
        # Showdown formats messages as multiline strings, optionally with a room prefix
        # Example: ">roomname\n|c:|...\n|c|..."
        lines = raw_msg.splitlines()
        room = ""
        if lines and lines[0].startswith(">"):
            room = lines[0][1:]
            lines = lines[1:]

        for line in lines:
            if not line.strip():
                continue
            
            parts = line.split("|")
            if len(parts) > 1:
                cmd = parts[1]
                if cmd == "challstr":
                    # Handle login challenge
                    challstr_id = parts[2]
                    challstr_key = parts[3]
                    asyncio.create_task(self._login(challstr_id, challstr_key))
                elif cmd == "updateuser":
                    # Check if login succeeded
                    logged_user = parts[2]
                    named = parts[3]
                    if named == "1":
                        logger.info(f"Successfully logged in as {logged_user}.")
                        # Join the configured room
                        await self.send(f"|/join {self.room}")
            
            # Put the message in the receive queue for the tracker/client to consume
            await self.receive_queue.put((room, line))

    async def _login(self, challstr_id: str, challstr_key: str):
        logger.info(f"Attempting to login to Showdown login server for user {self.username}...")
        challstr = f"{challstr_id}|{challstr_key}"
        
        loop = asyncio.get_running_loop()
        try:
            # Execute login HTTP POST synchronously in executor to avoid blocking event loop
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    self.login_url,
                    data={
                        "act": "login",
                        "name": self.username,
                        "pass": self.password,
                        "challstr": challstr,
                    }
                )
            )
            
            if response.status_code == 200:
                text = response.text.strip()
                if text.startswith(']'):
                    text = text[1:]
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    logger.error(f"Login assertion response was not JSON: {text!r}")
                    return

                if data.get("actionsuccess") and "assertion" in data:
                    assertion = data["assertion"]
                    # Send authentication command to Showdown websocket
                    await self.send(f"|/trn {self.username},0,{assertion}")
                else:
                    logger.error(f"Login assertion failed. Response: {data}")
            else:
                logger.error(f"HTTP request failed with status: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error during login assertion request: {e}")
