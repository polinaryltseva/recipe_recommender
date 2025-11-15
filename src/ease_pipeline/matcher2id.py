"""
Vkusvill Mapper - Convert local IDs to Vkusvill IDs
Provides utilities for mapping between local model IDs and Vkusvill product IDs
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class VkussvillMapper:
    """
    Maps between local model IDs and Vkusvill product IDs.

    Usage:
        mapper = VkussvillMapper('src/ease_pipeline/data/cooker2vkussvil.json')

        # Convert local IDs to Vkusvill IDs
        vkusvill_ids = mapper.local_to_vkusvill([0, 1, 2, 3])

        # Convert Vkusvill IDs back to local IDs
        local_ids = mapper.vkusvill_to_local([4862, 2964, 8964])

        # Check if local ID exists
        if mapper.has_local_id(5):
            vkusvill_id = mapper.get_vkusvill_id(5)
    """

    def __init__(self, mapping_path: str):
        """
        Initialize the mapper with local to Vkusvill ID mappings.

        Args:
            mapping_path: Path to JSON file with local_id->vkusvill_id mappings
        """
        self.mapping_path = Path(mapping_path)

        # Load mappings
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.local2vkusvill = json.load(f)

        # Convert string keys to int
        self.local2vkusvill: Dict[int, int] = {
            int(k): v for k, v in self.local2vkusvill.items()
        }

        # Create reverse mapping: vkusvill_id -> local_id
        self.vkusvill2local: Dict[int, int] = {
            v: k for k, v in self.local2vkusvill.items()
        }

        print(
            f"Loaded {len(self.local2vkusvill)} ID mappings from {self.mapping_path.name}"
        )

    def local_to_vkusvill(
        self, local_ids: List[int], skip_unknown: bool = True, warn_unknown: bool = True
    ) -> List[int]:
        """
        Convert local IDs to Vkusvill IDs.

        Args:
            local_ids: List of local model IDs
            skip_unknown: If True, skip unknown IDs; if False, raise error
            warn_unknown: If True, print warnings for unknown IDs

        Returns:
            List of Vkusvill product IDs

        Raises:
            ValueError: If skip_unknown=False and unknown ID found
        """
        vkusvill_ids = []

        for local_id in local_ids:
            if local_id in self.local2vkusvill:
                vkusvill_ids.append(self.local2vkusvill[local_id])
            else:
                if warn_unknown:
                    print(f"Warning: Local ID '{local_id}' not found in mappings")
                if not skip_unknown:
                    raise ValueError(f"Unknown local ID: '{local_id}'")

        return vkusvill_ids

    def vkusvill_to_local(
        self, vkusvill_ids: List[int], default_format: str = "Unknown_{id}"
    ) -> List[int]:
        """
        Convert Vkusvill IDs to local IDs.

        Args:
            vkusvill_ids: List of Vkusvill product IDs
            default_format: Format string for unknown IDs (use {id} placeholder)

        Returns:
            List of local model IDs
        """
        local_ids = []

        for vkusvill_id in vkusvill_ids:
            if vkusvill_id in self.vkusvill2local:
                local_ids.append(self.vkusvill2local[vkusvill_id])
            else:
                # For unknown IDs, we could either skip or use a placeholder
                # Here we skip them to maintain consistency
                pass

        return local_ids

    def get_vkusvill_id(self, local_id: int) -> Optional[int]:
        """
        Get Vkusvill ID for a single local ID.

        Args:
            local_id: Local model ID

        Returns:
            Vkusvill product ID or None if not found
        """
        return self.local2vkusvill.get(local_id)

    def get_local_id(self, vkusvill_id: int) -> Optional[int]:
        """
        Get local ID for a single Vkusvill ID.

        Args:
            vkusvill_id: Vkusvill product ID

        Returns:
            Local model ID or None if not found
        """
        return self.vkusvill2local.get(vkusvill_id)

    def has_local_id(self, local_id: int) -> bool:
        """Check if local ID exists in mappings."""
        return local_id in self.local2vkusvill

    def has_vkusvill_id(self, vkusvill_id: int) -> bool:
        """Check if Vkusvill ID exists in mappings."""
        return vkusvill_id in self.vkusvill2local

    def get_all_local_ids(self) -> List[int]:
        """Get list of all available local IDs."""
        return sorted(self.local2vkusvill.keys())

    def get_all_vkusvill_ids(self) -> List[int]:
        """Get list of all available Vkusvill IDs."""
        return sorted(self.vkusvill2local.keys())


def load_vkusvill_mapper(
    mapping_path: str,
) -> VkussvillMapper:
    """
    Convenience function to load the Vkusvill ID mapper.

    Args:
        mapping_path: Path to the local-to-vkusvill mapping JSON file

    Returns:
        Initialized VkussvillMapper
    """
    return VkussvillMapper(mapping_path)
