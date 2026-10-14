# Examples

Client examples for the running server: model `qwen3.8-flash-next`, base URL
`http://127.0.0.1:8000/v1`, no API key unless you set `API_KEY` in `.env`
(then export `API_KEY` before running these). Set `BASE_URL` to reach a
server on another machine (set `HOST` and `API_KEY` in `.env` on the server, see the README).

| File | What it shows |
|---|---|
| [`chat.sh`](chat.sh) | one chat request with `curl` and `jq` |
| [`chat.py`](chat.py) | the same with the OpenAI Python client, reasoning and answer separately |
| [`stream.py`](stream.py) | streaming, with reasoning and answer tokens told apart |
| [`tool_call.py`](tool_call.py) | a full tool-call round trip |

The Python examples need `pip install openai`.
