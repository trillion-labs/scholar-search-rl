import dataclasses
import json
import logging
from pathlib import Path

import tyro
from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DownloadArgs:
    dataset: str = "AlgorithmicResearchGroup/s2orc-cs-enriched"
    out_dir: Path = Path("data/raw/s2orc")
    config: str | None = None
    split: str = "train"
    allow_patterns: tuple[str, ...] = ("*.parquet", "README*", "*.json")
    max_workers: int = 64
    revision: str | None = None


def main(args: DownloadArgs) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        local_dir=str(args.out_dir),
        allow_patterns=list(args.allow_patterns),
        max_workers=args.max_workers,
        revision=args.revision,
    )

    parquet_files = sorted(Path(local_path).rglob("*.parquet"))
    total_bytes = sum(p.stat().st_size for p in parquet_files)

    manifest = {
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "config": args.config,
        "local_path": str(local_path),
        "num_parquet_files": len(parquet_files),
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / (1024**3), 2),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    log.info("downloaded %d parquet files (%.2f GB) to %s",
             len(parquet_files), manifest["total_gb"], local_path)
    log.info("manifest -> %s", manifest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    main(tyro.cli(DownloadArgs))
