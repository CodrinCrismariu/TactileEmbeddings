"""Read the FTP-1 Unit_Bimanual Zarr groups directly from their verified tar.

Extracting the 60 GB archive would duplicate storage. An existing index maps
each tar member name to (data_offset, byte_count), allowing Zarr to request
only the compressed chunks it needs for a clip. The store is read-only and
opens the archive for each lookup, so DataLoader workers need no shared handle.
"""

import json
from collections.abc import MutableMapping
from pathlib import Path


class IndexedTarZarrStore(MutableMapping):
    """Expose indexed tar-member bytes through Zarr's mapping interface.

    Inputs are the existing archive, its member-offset index and a group prefix.
    Reads return compressed member bytes; mutation is deliberately unsupported.
    """

    def __init__(self, tar_path, index_path, prefix):
        self.tar_path = Path(tar_path)
        index = json.loads(Path(index_path).read_text())
        signature = index["signature"]
        if self.tar_path.stat().st_size != signature[1]:
            raise ValueError("FTP tar size differs from the index signature")
        self.prefix = prefix.rstrip("/") + "/"
        self.members = {
            name[len(self.prefix) :]: tuple(location)
            for name, location in index["members"].items()
            if name.startswith(self.prefix) and not name.endswith("/")
        }
        if not self.members:
            raise ValueError(f"No Zarr members found under {self.prefix}")

    def __getitem__(self, key):
        try:
            offset, length = self.members[key]
        except KeyError:
            raise KeyError(key) from None
        with self.tar_path.open("rb") as stream:
            stream.seek(offset)
            data = stream.read(length)
        if len(data) != length:
            raise IOError(f"Truncated FTP tar member: {key}")
        return data

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)

    def __setitem__(self, key, value):
        raise TypeError("FTP tar store is read-only")

    def __delitem__(self, key):
        raise TypeError("FTP tar store is read-only")
