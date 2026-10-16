# Third-party notices

## ComfyUI MiniMax H3 Director

Parts of the Studio timeline interaction are adapted from the locally installed
`ComfyUI_MiniMaxH3_Director` project by AIMixer, notably its multi-segment
timeline design and source-video segment interaction model.

Source: `../ComfyUI/custom_nodes/ComfyUI_MiniMaxH3_Director/`

License: Apache License 2.0. The upstream `LICENSE` is retained at that source
location. Adapted code in Studio is marked with an in-source attribution.

## LangGraph and Toonflow architecture reference

The product assistant uses LangGraph as an installed Python dependency
(MIT license, https://github.com/langchain-ai/langgraph). Its separation of
model, tools, resources, and project context is inspired by Toonflow-app
(MIT license, https://github.com/HBAI-Ltd/Toonflow-app). No Toonflow agent
runtime code was copied into Studio.
