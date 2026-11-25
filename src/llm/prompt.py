"""
Prompt Manager & Frozen Hash Verifier (D-19 & D-22).
Guarantees byte-level fidelity to frozen prompt v1 and v2 specifications.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FROZEN_V1_COMBINED_SHA256 = "234207cb7c36d10e07d1bee66da897c4378ec7bd88eeddebba1cda32d211a911"
FROZEN_V2_COMBINED_SHA256 = "e67d6cb3b1b546ce3cd2441f829f611be878553d632d9dd27bf372ba8797d8d3"
FROZEN_COMBINED_SHA256 = FROZEN_V2_COMBINED_SHA256

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts"


class FrozenPromptLoader:
    def __init__(self, version: str = "v2", prompt_dir: Path = PROMPT_DIR):
        self.version = version
        self.prompt_dir = prompt_dir
        self.system_path = prompt_dir / f"prompt_{version}.system.txt"
        self.user_path = prompt_dir / f"prompt_{version}.user.txt"
        self.schema_path = prompt_dir / f"prompt_{version}.schema.json"
        
        self.system_text = ""
        self.user_template = ""
        self.schema_json = ""
        self.combined_hash = ""
        self._load_and_verify()

    def _load_and_verify(self) -> None:
        if not self.system_path.exists() or not self.user_path.exists() or not self.schema_path.exists():
            raise FileNotFoundError(f"Frozen prompt files missing for {self.version} under: {self.prompt_dir}")

        self.system_text = self.system_path.read_text(encoding="utf-8")
        self.user_template = self.user_path.read_text(encoding="utf-8")
        self.schema_json = self.schema_path.read_text(encoding="utf-8")

        # Compute combined hash: sha256(system ‖ NUL ‖ user ‖ NUL ‖ schema)
        combined_payload = (
            self.system_text + chr(0) + self.user_template + chr(0) + self.schema_json
        ).encode("utf-8")
        self.combined_hash = hashlib.sha256(combined_payload).hexdigest()

        expected = FROZEN_V1_COMBINED_SHA256 if self.version == "v1" else FROZEN_V2_COMBINED_SHA256
        if self.combined_hash != expected:
            raise ValueError(
                f"Prompt integrity check failed for {self.version}!\n"
                f"Expected: {expected}\n"
                f"Got:      {self.combined_hash}\n"
                f"Prompt text must NEVER be edited without minting a new version."
            )

    def render_user_prompt(self, canonical_url: str) -> str:
        """Render user prompt with canonical URL."""
        return self.user_template.replace("{canonical_url}", canonical_url)
