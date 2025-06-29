import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class CorpCodeCache:
    """Manages a local JSON cache for storing and retrieving corporation codes."""

    def __init__(self, cache_path: str = 'cache/corp_code_cache.json'):
        """
        Initializes the cache manager.

        Args:
            cache_path (str): The file path for the JSON cache.
        """
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        self._load_cache()

    def _load_cache(self):
        """Loads the cache from the JSON file if it exists."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Successfully loaded {len(self.cache)} corp codes from cache.")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load corp code cache from {self.cache_path}: {e}")
                self.cache = {}
        else:
            logger.info("Corp code cache file not found. A new one will be created.")

    def get(self, identifier: str) -> Optional[str]:
        """Retrieves a corp code from the cache."""
        return self.cache.get(identifier)

    def set(self, identifier: str, corp_code: str):
        """
        Adds a new corp code to the cache and saves it to the file.
        """
        if identifier and corp_code:
            self.cache[identifier] = corp_code
            self._save_cache()

    def _save_cache(self):
        """
        Saves the current cache state to the JSON file.
        """
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=4)
        except IOError as e:
            logger.error(f"Could not save corp code cache to {self.cache_path}: {e}")
