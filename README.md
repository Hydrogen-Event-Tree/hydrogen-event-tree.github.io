<p align="center">
  <img src="screenshot1.png" width="48.4%"/>
  <img src="screenshot2.png" width="50.6%"/>
</p>

# H2 Event Tree

This is the repository for creating the dashboard at <a href="https://hydrogen-event-tree.github.io/" target="_blank">https://hydrogen-event-tree.github.io/</a> where we present a configurable event tree based on events from the
The Hydrogen Incident and Accidents Database <a href="https://minerva.jrc.ec.europa.eu/en/shorturl/capri/hiadpt" target="_blank">HIAD 2.1</a>, developed by The Joint Research Centre (JRC) of the European Commission. This project was developed as a part of <a href="https://hydrogeni.no/" target="_blank">HYDROGENi</a>, a Norwegian research and innovation centre for hydrogen and ammonia. More details are available in the paper (link coming soon) which this dashboard is a part of.

Obviously, we provide no guarantees about the correctness of any of the information on the dashboard but rather present it as our best effort at creating a useful event tree from the events in HIAD. Feel free to submit PRs or open issues if you feel anything should be changed, or fork a version yourself.

## Implementation

The dashboard loads `events-manifest.json` in the same top-level folder as `index.html`. The manifest lists one or more model entries:

```json
{
  "models": [
    {
      "id": "qwen3-14b",
      "name": "Qwen3-14B",
      "events_path": "events-qwen3-14b.json",
      "model": "qwen3:14b"
    }
  ],
  "default_model_id": "qwen3-14b"
}
```

Each `events_path` points to a JSON array produced by `main.py`. To generate (or refresh) these files, set `gen = 1` in `main.py`. It will run `process_events()` for every entry in `LLMS`, write `events-<id>.json` files, and rebuild `events-manifest.json`. If you already have the JSON files, leave `gen = 0` and the script will just start the webserver on `localhost:4000`.

### Models

Models are configured in `LLMS` at the top of `main.py`. For local inference, `ollama` must be installed both as a pip package and as a running service (https://ollama.com/). For OpenRouter-backed entries, set `OPENROUTER_API_KEY` in `main.py` before running the generator, and ensure the `openai` package is installed.
