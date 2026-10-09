import os
import itertools
from threading import Lock

class KeyManager:
    def __init__(self, env_var_name: str):
        keys_str = os.getenv(env_var_name, "")
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        self.cycle = itertools.cycle(self.keys) if self.keys else None
        self.lock = Lock()
        
    def get_next_key(self):
        if not self.keys:
            return None
        with self.lock:
            return next(self.cycle)

gemini_keys = KeyManager("GOOGLE_API_KEY")
groq_keys = KeyManager("GROQ_API_KEY")
