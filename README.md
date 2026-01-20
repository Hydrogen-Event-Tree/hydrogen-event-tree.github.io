<p align="center">
  <img src="screenshot1" width="48.4%"/>
  <img src="screenshot2" width="50.6%"/>
</p>

# H2 Event Tree

This is the repository for creating the dashboard at <a href="https://simonhalvdansson.github.io/H2-Event-Tree/" target="_blank">https://simonhalvdansson.github.io/H2-Event-Tree/</a> where we present a configurable event tree based on events from the
The Hydrogen Incident and Accidents Database <a href="https://minerva.jrc.ec.europa.eu/en/shorturl/capri/hiadpt" target="_blank">HIAD 2.1</a>, developed by The Joint Research Centre (JRC) of the European Commission. This project was developed as a part of <a href="https://hydrogeni.no/" target="_blank">HYDROGENi</a>, a Norwegian research and innovation centre for hydrogen and ammonia. More details are available in the paper (link coming soon) which this dashboard is a part of.

Obviously, we provide no guarantees about the correctness of any of the information on the dashboard but rather present it as our best effort at creating a useful event tree from the events in HIAD. Feel free to submit PRs or open issues if you feel anything should be changed, or fork a version yourself.

## Implementation

The dashboard is dependent on the existence of a `events.json` file in the same top-level folder as `index.html`. To generate this file, we run `main.py` which can call `process_events()` to read a `hiad.xlsx` file in the same folder and produce the `events.json` file. This function call can be skipped if the file is already in place. In that case, the script spins up a webserver on `localhost:4000` to serve the dashboard.

To process the Excel sheet with `process_events()`, `ollama` needs to be installed. Both as a pip package and also as a service running on your computer (https://ollama.com/). Download a model and indicate it in the `get_llm()` functionn at the top of `main.py`.
