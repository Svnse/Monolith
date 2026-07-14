---
name: soundtrap
description: Manage the local Soundtrap loop workspace: projects, audio clips, placements, and generated clip requests.
---

Examples:

{"tool":"soundtrap","op":"state"}
{"tool":"soundtrap","op":"create_project","name":"Night Loop","bpm":120}
{"tool":"soundtrap","op":"set_bpm","bpm":128}
{"tool":"soundtrap","op":"add_track","name":"drums"}
{"tool":"soundtrap","op":"add_clip","path":"C:/path/kick.wav","name":"Kick"}
{"tool":"soundtrap","op":"place_clip","clip_id":"clip_123","track":"drums","start_beat":0}
{"tool":"soundtrap","op":"move_placement","placement_id":"placement_123","track":"drums","start_beat":4,"length_beats":2}
{"tool":"soundtrap","op":"remove_placement","placement_id":"placement_123"}
{"tool":"soundtrap","op":"generate_clip","prompt":"short warm analog bass loop","duration":6}
