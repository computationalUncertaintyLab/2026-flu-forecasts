#!/usr/bin/env python3

"""
Download the WHO FluMart VIW_FID_EPI dataset as a CSV file.

Usage
-----
python download_who_flumart.py

python download_who_flumart.py \
    --output data/who_viw_fid_epi.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://xmart-api-public.who.int/FLUMART/VIW_FID_EPI"


def create_session() -> requests.Session:
    """Create a requests session with automatic retry behavior."""

    retry_strategy = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=2,
        pool_maxsize=2,
    )

    session = requests.Session()
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": "WHO-FluMart-Downloader/1.0",
            "Accept": "text/csv, application/csv, */*",
            "Accept-Encoding": "gzip, deflate",
        }
    )

    return session


def format_bytes(n_bytes: int) -> str:
    """Return a human-readable file size."""

    value = float(n_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= 1024

    return f"{value:,.1f} TB"


def download_dataset(
    output_path: Path,
    chunk_size: int = 1024 * 1024,
) -> None:
    """
    Stream the WHO FluMart CSV export to disk.

    The file is first written to a temporary file and moved into place only
    after the download completes successfully. This avoids leaving a partial
    CSV with the final output filename.
    """

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "$format": "csv",
    }

    session = create_session()

    temporary_path: Path | None = None

    try:
        print(f"Downloading:\n  {BASE_URL}")
        print(f"Saving to:\n  {output_path}")

        with session.get(
            BASE_URL,
            params=params,
            stream=True,
            timeout=(30, 600),
        ) as response:

            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            content_length = response.headers.get("Content-Length")

            if content_length is not None:
                expected_size = int(content_length)
                print(f"Expected download size: {format_bytes(expected_size)}")
            else:
                expected_size = None
                print("The server did not report the total download size.")

            if "csv" not in content_type.lower():
                print(
                    f"Warning: server returned Content-Type {content_type!r}.",
                    file=sys.stderr,
                )

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"{output_path.name}.",
                suffix=".part",
                dir=output_path.parent,
                delete=False,
            ) as temporary_file:

                temporary_path = Path(temporary_file.name)
                bytes_downloaded = 0
                next_progress_update = 25 * 1024 * 1024

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue

                    temporary_file.write(chunk)
                    bytes_downloaded += len(chunk)

                    if bytes_downloaded >= next_progress_update:
                        if expected_size:
                            percent = 100 * bytes_downloaded / expected_size
                            print(
                                f"\rDownloaded "
                                f"{format_bytes(bytes_downloaded)} "
                                f"({percent:.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\rDownloaded "
                                f"{format_bytes(bytes_downloaded)}",
                                end="",
                                flush=True,
                            )

                        next_progress_update += 25 * 1024 * 1024

                temporary_file.flush()

        if bytes_downloaded == 0:
            raise RuntimeError("The server returned an empty response.")

        if expected_size is not None and bytes_downloaded != expected_size:
            raise RuntimeError(
                "Downloaded file size does not match the server's "
                f"Content-Length: received {bytes_downloaded:,} bytes, "
                f"expected {expected_size:,} bytes."
            )

        # Quick check for an HTML or JSON error response saved as a CSV.
        with temporary_path.open("rb") as file:
            beginning = file.read(500).lstrip().lower()

        if beginning.startswith(b"<!doctype html") or beginning.startswith(
            b"<html"
        ):
            raise RuntimeError(
                "The server returned an HTML page rather than a CSV file."
            )

        if beginning.startswith(b"{") and b"error" in beginning:
            raise RuntimeError(
                "The server appears to have returned a JSON error response."
            )

        shutil.move(str(temporary_path), str(output_path))
        temporary_path = None

        print()
        print(
            f"Download complete: {output_path} "
            f"({format_bytes(bytes_downloaded)})"
        )

    except requests.HTTPError as error:
        status_code = (
            error.response.status_code if error.response is not None else None
        )
        raise RuntimeError(
            f"WHO server returned HTTP status {status_code}."
        ) from error

    except requests.RequestException as error:
        raise RuntimeError(
            f"Unable to download the WHO dataset: {error}"
        ) from error

    finally:
        session.close()

        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the WHO FluMart VIW_FID_EPI dataset."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/who_viw_fid_epi.csv"),
        help=(
            "Destination CSV path. "
            "Default: who_viw_fid_epi.csv"
        ),
    )

    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=1,
        help="Download chunk size in megabytes. Default: 1",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.chunk_size_mb < 1:
        print("--chunk-size-mb must be at least 1.", file=sys.stderr)
        return 2

    try:
        download_dataset(
            output_path=args.output,
            chunk_size=args.chunk_size_mb * 1024 * 1024,
        )
    except RuntimeError as error:
        print(f"\nDownload failed: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
