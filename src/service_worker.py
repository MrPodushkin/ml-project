"""One persistent model process, with a hard deadline and explicit reaping."""

import asyncio
import multiprocessing
from pathlib import Path


def serve_model(connection, model_path, manifest):
    from src.model_utils import load_model
    from src.predict_sms import predict_sms

    try:
        model = load_model(Path(model_path), Path(manifest))
        connection.send({"ready": True})
    except Exception:
        connection.send({"ready": False})
        connection.close()
        return
    try:
        while True:
            message = connection.recv()
            if message is None:
                break
            try:
                result = predict_sms(message, model)
                connection.send({"prediction": result})
            except Exception:
                connection.send({"error": "prediction_failed"})
    except EOFError, BrokenPipeError:
        pass
    finally:
        connection.close()


class ModelWorker:
    def __init__(self, model_path, manifest, timeout=2.0, startup_timeout=15.0):
        self.model_path = str(model_path)
        self.manifest = str(manifest)
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.process = None
        self.connection = None
        self.ready = False
        self.calls = 0
        self.generation = 0
        self.lock = asyncio.Lock()
        self.recovery = None

    @property
    def healthy(self):
        return bool(self.ready and self.process and self.process.is_alive())

    def start(self):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=serve_model, args=(child, self.model_path, self.manifest), daemon=True
        )
        self.process.start()
        child.close()
        self.generation += 1
        try:
            self.ready = self.connection.poll(self.startup_timeout) and bool(
                self.connection.recv().get("ready")
            )
        except EOFError, OSError:
            self.ready = False
        if not self.ready:
            self.stop()

    def stop(self):
        self.ready = False
        if self.process:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=0.25)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1)
            self.process.close()
            self.process = None
        if self.connection:
            self.connection.close()
            self.connection = None

    def exchange(self, message):
        self.calls += 1
        self.connection.send(message)
        if not self.connection.poll(self.timeout):
            raise TimeoutError("prediction_timeout")
        return self.connection.recv()

    async def recover(self):
        await asyncio.to_thread(self.start)

    async def predict(self, message):
        # No queue: one inference slot; excess requests are rejected by the API.
        if self.lock.locked():
            raise BlockingIOError("inference_busy")
        async with self.lock:
            if not self.healthy:
                raise ConnectionError("model_unavailable")
            try:
                result = await asyncio.to_thread(self.exchange, message)
            except TimeoutError, EOFError, OSError:
                await asyncio.to_thread(self.stop)
                self.recovery = asyncio.create_task(self.recover())
                raise TimeoutError("prediction_timeout") from None
            if "error" in result:
                raise RuntimeError("prediction_failed")
            return result["prediction"]

    async def close(self):
        if self.recovery:
            await self.recovery
        await asyncio.to_thread(self.stop)
