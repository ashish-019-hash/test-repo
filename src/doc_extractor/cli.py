"""`doc-extractor` command line interface.

Exit codes
----------
0  success
2  a pipeline stage failed after retries (message names the stage and the resume command)
3  invalid input, configuration or usage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from doc_extractor import __version__
from doc_extractor.agents import AGENT_REGISTRY
from doc_extractor.exceptions import ConfigError, DocExtractorError, IngestError
from doc_extractor.observability import configure_logging, get_logger
from doc_extractor.schemas.state import STAGE_ORDER

EXIT_OK = 0
EXIT_STAGE_FAILED = 2
EXIT_USAGE = 3

_PROVIDERS = ("auto", "azure", "rules", "mock")
_STAGE_NAMES = tuple(str(s) for s in STAGE_ORDER)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, default=None, help="YAML config file (merged over defaults)")
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=".env file to load (default: nearest .env found from the current directory)",
    )
    p.add_argument(
        "--provider",
        choices=_PROVIDERS,
        default=None,
        help="LLM provider; overrides llm.provider from config (auto = Azure if configured, else rules)",
    )
    p.add_argument("--log-format", choices=("console", "json"), default=None)
    p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-extractor",
        description="Multi-agent document attribute and entity extraction (LangGraph + Azure OpenAI).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the pipeline on one document")
    run.add_argument("file", type=Path, help="Input document (.pdf, .docx, .md, .txt)")
    run.add_argument("--out", type=Path, default=Path("out"), help="Output directory (default: out/)")
    run.add_argument("--resume", action="store_true", help="Continue after the last checkpointed stage")
    run.add_argument(
        "--resume-from",
        choices=_STAGE_NAMES,
        default=None,
        metavar="STAGE",
        help=f"Re-run from STAGE using the stored snapshot of its predecessor. One of: {', '.join(_STAGE_NAMES)}",
    )
    run.add_argument(
        "--graph",
        choices=("json", "graphml"),
        default=None,
        help="Also export the attribute/entity graph in this format",
    )
    _add_common(run)

    stages = sub.add_parser("stages", help="List pipeline stages in order")
    stages.add_argument("--json", action="store_true", help="Emit JSON instead of a table")

    config = sub.add_parser("config", help="Inspect the effective configuration")
    config.add_argument("--show", action="store_true", help="Print the effective config and its hash")
    _add_common(config)
    return parser


def _load_env(args: argparse.Namespace) -> None:
    from doc_extractor.llm.settings import load_dotenv_file

    env_file = getattr(args, "env_file", None)
    if env_file is not None and not Path(env_file).is_file():
        raise ConfigError(f"--env-file not found: {env_file}")
    load_dotenv_file(str(env_file) if env_file else None)


def _load_cfg(args: argparse.Namespace) -> Any:
    from doc_extractor.config import load_config

    env = dict(os.environ)
    if getattr(args, "provider", None):
        env["LLM_PROVIDER"] = args.provider
    cfg = load_config(getattr(args, "config", None), env=env)
    updates: dict[str, Any] = {}
    if getattr(args, "graph", None):
        updates["output"] = cfg.output.model_copy(update={"graph_export": args.graph})
    if getattr(args, "log_level", None) or getattr(args, "log_format", None):
        updates["logging"] = cfg.logging.model_copy(
            update={
                k: v for k, v in {"level": args.log_level, "format": args.log_format}.items() if v is not None
            }
        )
    return cfg.model_copy(update=updates) if updates else cfg


def cmd_stages(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = [
        {
            "index": i,
            "stage": str(stage),
            "agent": cls.__name__,
            "requires": list(cls.requires),
            "produces": list(cls.produces),
        }
        for i, (stage, cls) in enumerate(AGENT_REGISTRY.items(), start=1)
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    width = max(len(str(r["stage"])) for r in rows)
    for r in rows:
        requires = ", ".join(str(x) for x in r["requires"]) or "-"
        produces = ", ".join(str(x) for x in r["produces"]) or "-"
        print(f"{r['index']:>2}. {str(r['stage']):<{width}}  {r['agent']}")
        print(f"    requires: {requires}")
        print(f"    produces: {produces}")
    return EXIT_OK


def cmd_config(args: argparse.Namespace) -> int:
    _load_env(args)
    cfg = _load_cfg(args)
    from doc_extractor.storage.canonical_json import dumps

    payload = cfg.model_dump(mode="json")
    payload["config_hash"] = cfg.config_hash
    payload["lexicon_paths"] = {k: str(v) for k, v in sorted(cfg.lexicon_paths.items())}
    print(dumps(payload), end="")
    return EXIT_OK


def cmd_run(args: argparse.Namespace) -> int:
    _load_env(args)
    cfg = _load_cfg(args)
    configure_logging(level=cfg.logging.level, fmt=cfg.logging.format)
    log = get_logger()
    if args.resume and args.resume_from:
        raise ConfigError("--resume and --resume-from are mutually exclusive")

    from doc_extractor.graph.runner import run_pipeline

    result = run_pipeline(
        args.file,
        args.out,
        cfg,
        resume=args.resume,
        resume_from=args.resume_from,
        log=log,
    )
    if not result.ok:
        print(
            f"Stage '{result.failed_stage}' failed: {result.error}\n"
            f"Fix the cause, then continue with:\n"
            f"  doc-extractor run {args.file} --out {args.out} --resume\n"
            f"or re-run that stage from its stored snapshot with:\n"
            f"  doc-extractor run {args.file} --out {args.out} --resume-from {result.failed_stage}",
            file=sys.stderr,
        )
        return EXIT_STAGE_FAILED
    final = result.state.get("final_output")
    n_attr = len(final.attributes) if final else 0
    n_ent = len(final.entities) if final else 0
    n_map = len(final.mappings) if final else 0
    n_review = len(result.state.get("human_review_queue", []))
    print(
        f"OK  provider={result.metadata.provider}  attributes={n_attr}  entities={n_ent}  "
        f"mappings={n_map}  review_items={n_review}\n"
        f"Outputs written to {result.out_dir}/ (final.json, entities.json, mappings.json, "
        f"attributes.json, discarded.json, review_queue.json, run.json)"
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {"run": cmd_run, "stages": cmd_stages, "config": cmd_config}
    try:
        return handlers[args.command](args)
    except (ConfigError, IngestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except DocExtractorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_STAGE_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
