# engine/_workers — subprocess worker entry points
# Each module here runs inside a child process spawned by an EngineProcess.
# Heavy imports (torch, diffusers, whisper, etc.) live only in these files.
