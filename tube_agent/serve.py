"""Minimal local Chat Completions endpoint. Uses Transformers, no vLLM required."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os

from .model_utils import DEFAULT_MODEL, resolve_model


def convert_messages(messages):
    from PIL import Image
    converted = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append(part)
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/"):
                        raise ValueError("Offline server accepts embedded images only")
                    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
                    parts.append({"type": "image", "image": image})
                else:
                    raise ValueError("Unsupported message content")
            content = parts
        else:
            raise ValueError("Message content must be text or a list of content parts")
        converted.append({"role": message["role"], "content": content})
    return converted


def build_inputs(processor, messages):
    """Use the tokenizer's template explicitly so enable_thinking is respected."""
    messages = convert_messages(messages)
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    images = [part["image"] for message in messages for part in message["content"]
              if part["type"] == "image"]
    if images:
        return processor(text=[prompt], images=images, return_tensors="pt")
    return processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", help="Optional trained LoRA directory")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-context", type=int, default=8192)
    args = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("Run the model server on a GPU instance; simulator-only checks work on CPU")
    path = str(resolve_model(args.model))
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa", local_files_only=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"ready": True, "model": "qwen-tube", "adapter": args.adapter})
            else:
                self.respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.respond(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32*1024*1024:
                    raise ValueError("Invalid request size")
                request = json.loads(self.rfile.read(length))
                inputs = build_inputs(processor, request["messages"])
                inputs = inputs.to(model.device)
                size = inputs["input_ids"].shape[1]
                max_tokens = min(int(request.get("max_tokens", 192)), 512)
                if max_tokens < 1 or size + max_tokens > args.max_context:
                    raise ValueError(f"Request exceeds context limit {args.max_context}, or invalid max_tokens")
                temperature = float(request.get("temperature", 0.))
                torch.manual_seed(int(request.get("seed", 0)))
                options = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
                if temperature > 0:
                    options.update(temperature=temperature, top_p=.95)
                with torch.inference_mode():
                    output = model.generate(**inputs, **options)
                answer = processor.decode(output[0, size:], skip_special_tokens=True)
                self.respond(200, {"choices": [{"message": {"role": "assistant", "content": answer}}],
                                   "usage": {"prompt_tokens": size, "completion_tokens": int(output.shape[1]-size)}})
            except Exception as exc:
                self.respond(400, {"error": f"{type(exc).__name__}: {exc}"})

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Ready: http://127.0.0.1:{args.port}/v1 ; model={path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
