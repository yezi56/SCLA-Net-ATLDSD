# Attention Plugins

Each file in this directory is a named entry point for a hot-pluggable feature
module. The shared implementations live in `src/modules/plugins/modules.py`, and
`src/modules/plugins/factory.py` maps `attention_type` strings to these classes.

Use `build_attention(name, channels)` from `plugins` in model code. Use these
files when you want to quickly find a specific module by name.
