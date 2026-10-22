"""Inspect FTP-1 tar/Zarr metadata using HTTP byte ranges, without downloading GBs.

This is a scouting utility: it prints arrays in the first Zarr group only.
It does not prove that every task in an archive has the same schema.
"""

import argparse
import json
import urllib.request


BASE = ("https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/resolve/"
        "d287c9c532514a08b2eb29bf1259997bb32f4f3b/")
BLOCK = 64 * 1024


class RemoteFile:
    def __init__(self, url):
        self.url = url
        self.cache = {}
        self.requests = 0

    def read(self, offset, size):
        result = bytearray()
        while len(result) < size:
            pos = offset + len(result)
            block_start = pos // BLOCK * BLOCK
            if block_start not in self.cache:
                req = urllib.request.Request(
                    self.url, headers={"Range": f"bytes={block_start}-{block_start + BLOCK - 1}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    self.url = response.geturl()
                    self.cache[block_start] = response.read()
                self.requests += 1
            block = self.cache[block_start]
            part = block[pos - block_start:pos - block_start + size - len(result)]
            if not part:
                raise EOFError(f"No data at byte {pos}")
            result.extend(part)
        return bytes(result)


def inspect(group, max_headers):
    remote = RemoteFile(f"{BASE}{group}/{group}.tar")
    offset = 0
    first_zarr = None
    arrays = {}
    for _ in range(max_headers):
        header = remote.read(offset, 512)
        name = header[:100].rstrip(b"\0").decode("utf8", "replace")
        prefix = header[345:500].rstrip(b"\0").decode("utf8", "replace")
        name = f"{prefix}/{name}" if prefix else name
        if not name:
            break
        size = int(header[124:136].strip(b"\0 ") or b"0", 8)
        if ".zarr/" in name:
            zarr_name = name.split(".zarr/", 1)[0] + ".zarr"
            if first_zarr is None:
                first_zarr = zarr_name
            elif zarr_name != first_zarr:
                break
            if "/data/" in name and name.endswith("/.zarray"):
                key = name.split("/data/", 1)[1].removesuffix("/.zarray")
                arrays[key] = json.loads(remote.read(offset + 512, size))
        offset += 512 + (size + 511) // 512 * 512
    print(json.dumps({"group": group, "first_zarr": first_zarr,
                      "array_shapes": {k: v["shape"] for k, v in arrays.items()},
                      "headers_scanned": _ + 1, "range_requests": remote.requests}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="+")
    parser.add_argument("--max-headers", type=int, default=500)
    args = parser.parse_args()
    for group in args.groups:
        inspect(group, args.max_headers)


if __name__ == "__main__":
    main()
