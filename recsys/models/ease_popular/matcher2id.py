"""
Vkusvill Mapper - Convert internal IDs to external Vkusvill IDs
Provides utilities for mapping between internal model IDs and external Vkusvill product IDs
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class VkussvillMapper:
    """
    Maps between internal model IDs (pov) and external Vkusvill product IDs (vv).

    Usage:
        mapper = VkussvillMapper(
            'src/ease_pipeline/data/items_dict_vv2pov_str.json',
            'src/ease_pipeline/data/items_dict_pov2vv_str.json'
        )

        # Convert external Vkusvill IDs to internal IDs (input)
        internal_ids = mapper.vkusvill_to_internal([4862, 2964, 8964])

        # Convert internal IDs back to external Vkusvill IDs (output)
        vkusvill_ids = mapper.internal_to_vkusvill([0, 1, 2, 3])

        # Check if internal ID exists
        if mapper.has_internal_id(5):
            vkusvill_id = mapper.get_vkusvill_id(5)
    """

    def __init__(self, vv2pov_path: str, pov2vv_path: str):
        """
        Initialize the mapper with bidirectional mappings.

        Args:
            vv2pov_path: Path to JSON file with vkusvill_id->internal_id mappings (for input)
            pov2vv_path: Path to JSON file with internal_id->vkusvill_id mappings (for output)
        """
        self.vv2pov_path = Path(vv2pov_path)
        self.pov2vv_path = Path(pov2vv_path)

        # Load vkusvill to internal mapping (for input)
        with open(self.vv2pov_path, "r", encoding="utf-8") as f:
            vv2pov_data = json.load(f)

        # Convert string keys and values to int
        self.vkusvill2internal: Dict[int, int] = {
            int(k): int(v) for k, v in vv2pov_data.items()
        }

        # Load internal to vkusvill mapping (for output)
        with open(self.pov2vv_path, "r", encoding="utf-8") as f:
            pov2vv_data = json.load(f)

        # Convert string keys to int, values are already strings representing IDs
        self.internal2vkusvill: Dict[int, int] = {
            int(k): int(v) for k, v in pov2vv_data.items()
        }

        # print(
        #     f"Loaded {len(self.vkusvill2internal)} vv->pov mappings from {self.vv2pov_path.name}"
        # )
        # print(
        #     f"Loaded {len(self.internal2vkusvill)} pov->vv mappings from {self.pov2vv_path.name}"
        # )

    def internal_to_vkusvill(
        self, internal_ids: List[int], skip_unknown: bool = True, warn_unknown: bool = True
    ) -> List[int]:
        """
        Convert internal IDs to external Vkusvill IDs (for output).

        Args:
            internal_ids: List of internal model IDs
            skip_unknown: If True, skip unknown IDs; if False, raise error
            warn_unknown: If True, print warnings for unknown IDs

        Returns:
            List of external Vkusvill product IDs

        Raises:
            ValueError: If skip_unknown=False and unknown ID found
        """
        vkusvill_ids = []

        for internal_id in internal_ids:
            if internal_id in self.internal2vkusvill:
                vkusvill_ids.append(self.internal2vkusvill[internal_id])
            else:
                if warn_unknown:
                    print(f"Warning: Internal ID '{internal_id}' not found in mappings")
                if not skip_unknown:
                    raise ValueError(f"Unknown internal ID: '{internal_id}'")

        return vkusvill_ids

    def vkusvill_to_internal(
        self, vkusvill_ids: List[int], skip_unknown: bool = True, warn_unknown: bool = True
    ) -> List[int]:
        """
        Convert external Vkusvill IDs to internal IDs (for input).

        Args:
            vkusvill_ids: List of external Vkusvill product IDs
            skip_unknown: If True, skip unknown IDs; if False, raise error
            warn_unknown: If True, print warnings for unknown IDs

        Returns:
            List of internal model IDs

        Raises:
            ValueError: If skip_unknown=False and unknown ID found
        """
        internal_ids = []

        for vkusvill_id in vkusvill_ids:
            if vkusvill_id in self.vkusvill2internal:
                internal_ids.append(self.vkusvill2internal[vkusvill_id])
            else:
                if warn_unknown:
                    print(f"Warning: Vkusvill ID '{vkusvill_id}' not found in mappings")
                if not skip_unknown:
                    raise ValueError(f"Unknown Vkusvill ID: '{vkusvill_id}'")
                # Skip unknown IDs when skip_unknown=True

        return internal_ids

    def get_vkusvill_id(self, internal_id: int) -> Optional[int]:
        """
        Get external Vkusvill ID for a single internal ID.

        Args:
            internal_id: Internal model ID

        Returns:
            External Vkusvill product ID or None if not found
        """
        return self.internal2vkusvill.get(internal_id)

    def get_internal_id(self, vkusvill_id: int) -> Optional[int]:
        """
        Get internal ID for a single external Vkusvill ID.

        Args:
            vkusvill_id: External Vkusvill product ID

        Returns:
            Internal model ID or None if not found
        """
        return self.vkusvill2internal.get(vkusvill_id)

    def has_internal_id(self, internal_id: int) -> bool:
        """Check if internal ID exists in mappings."""
        return internal_id in self.internal2vkusvill

    def has_vkusvill_id(self, vkusvill_id: int) -> bool:
        """Check if external Vkusvill ID exists in mappings."""
        return vkusvill_id in self.vkusvill2internal

    def get_all_internal_ids(self) -> List[int]:
        """Get list of all available internal IDs."""
        return sorted(self.internal2vkusvill.keys())

    def get_all_vkusvill_ids(self) -> List[int]:
        """Get list of all available external Vkusvill IDs."""
        return sorted(self.vkusvill2internal.keys())


def load_vkusvill_mapper(
    vv2pov_path: str = "src/ease_pipeline/data/items_dict_vv2pov_str.json",
    pov2vv_path: str = "src/ease_pipeline/data/items_dict_pov2vv_str.json",
) -> VkussvillMapper:
    """
    Convenience function to load the Vkusvill ID mapper.

    Args:
        vv2pov_path: Path to the vkusvill-to-internal mapping JSON file
        pov2vv_path: Path to the internal-to-vkusvill mapping JSON file

    Returns:
        Initialized VkussvillMapper
    """
    return VkussvillMapper(vv2pov_path, pov2vv_path)
