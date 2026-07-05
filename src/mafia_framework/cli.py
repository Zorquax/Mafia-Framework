import argparse
import asyncio
import sys
from pathlib import Path

from .data.aliases import import_aliases_from_json, list_player_aliases, remove_player_alias, set_player_alias
from .data.models import initialize_database
from .io.google_docs import fetch_published_google_doc
from .io.ingestion import ingest_google_doc, ingest_log
from .services.game_service import assign_player_role, get_session, remove_game, set_game_display_name
from .services.model_service import format_train_metrics, predict_from_text, train_model_from_db
from .services.game_service import resolve_flip_map
from .analysis.tells import FEATURE_NAMES, default_tells, aggregate_tells
from .models.feature_engineering import build_feature_dataframe


def command_init_db(args: argparse.Namespace) -> int:
    initialize_database(args.db)
    print(f"Created or updated database at {args.db}")
    return 0


def command_ingest(args: argparse.Namespace) -> int:
    raw_text = Path(args.file).read_text(encoding="utf-8")
    game_id = ingest_log(args.db, raw_text, source=args.file)
    print(f"Ingested game into {args.db} with id={game_id}")
    return 0


def command_ingest_url(args: argparse.Namespace) -> int:
    game_id = ingest_google_doc(args.db, args.url)
    print(f"Ingested published Google Doc into {args.db} with id={game_id}")
    return 0


def command_train(args: argparse.Namespace) -> int:
    output = args.output or ("data/model_d1.pkl" if args.mode == "day-one" else "data/model.pkl")
    try:
        result = train_model_from_db(
            args.db,
            output,
            day_one=args.mode == "day-one",
            exclude_neutral_games=args.exclude_neutral_games,
        )
    except ValueError as exc:
        print(exc)
        return 1

    metrics = format_train_metrics(result.bundle)
    print(f"Saved trained model to {output}")
    print(f"Mode: {result.mode}")
    print(f"Feature set version: {result.bundle.feature_set_version}")
    print(f"CV accuracy: {metrics.get('accuracy', '0.00%')}")
    print(f"CV log loss: {metrics.get('log_loss', '0.0000')}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    session = get_session(args.db, args.game_id)
    if session is None:
        print("No game found in the database.")
        return 1

    tell_results = aggregate_tells(session, default_tells())
    df = build_feature_dataframe(tell_results, feature_names=FEATURE_NAMES)

    print(f"Source: {session.source}")
    if session.game_id is not None:
        print(f"Game id: {session.game_id}")
    print(f"Players: {', '.join(session.players) if session.players else 'none'}")
    print("Flips:")
    for flip in session.flips:
        print(f"  {flip.player_name}: {flip.alignment}")
    if not session.flips:
        print("  none")

    print("\nFeature data:")
    print(df.to_string(index=False) if not df.empty else "  no feature rows available")

    resolved = resolve_flip_map(session)
    unknown = sorted(set(session.players) - set(resolved.keys()))
    print("\nPlayers without explicit flips (unknown):")
    for player_name in unknown or ["none"]:
        print(f"  {player_name}")
    return 0


def command_delete_game(args: argparse.Namespace) -> int:
    try:
        remove_game(args.db, args.game_id)
        print(f"Deleted game {args.game_id} from {args.db}")
        return 0
    except FileNotFoundError as exc:
        print(exc)
        return 1


def command_set_game_name(args: argparse.Namespace) -> int:
    try:
        set_game_display_name(args.db, args.game_id, args.name)
        print(f"Set display name for game {args.game_id} to {args.name!r}")
        return 0
    except FileNotFoundError as exc:
        print(exc)
        return 1


def command_set_alias(args: argparse.Namespace) -> int:
    set_player_alias(args.db, args.alias, args.canonical)
    print(f"Mapped alias {args.alias!r} -> canonical {args.canonical!r}")
    return 0


def command_remove_alias(args: argparse.Namespace) -> int:
    removed = remove_player_alias(args.db, args.alias)
    if removed:
        print(f"Removed alias {args.alias!r}")
        return 0
    print(f"Alias {args.alias!r} not found.")
    return 1


def command_list_aliases(args: argparse.Namespace) -> int:
    rows = list_player_aliases(args.db)
    if not rows:
        print("No player aliases defined.")
        return 0
    for alias, canonical in rows:
        print(f"{alias} -> {canonical}")
    return 0


def command_import_aliases(args: argparse.Namespace) -> int:
    count = import_aliases_from_json(args.db, args.file)
    print(f"Imported {count} alias mapping(s) from {args.file}")
    return 0


def command_set_role(args: argparse.Namespace) -> int:
    try:
        assign_player_role(args.db, args.game_id, args.player, args.alignment)
        print(f"Set role for {args.player} in game {args.game_id} to {args.alignment}")
        return 0
    except FileNotFoundError as exc:
        print(exc)
        return 1


def command_predict(args: argparse.Namespace) -> int:
    raw_text = Path(args.file).read_text(encoding="utf-8")
    predictions = predict_from_text(
        raw_text,
        args.model,
        source=args.file,
        db_path=args.db,
        day_one=args.mode == "day-one",
    )
    for row in predictions:
        print(f"{row.player_name}: {row.formatted}")
    return 0


def command_predict_url(args: argparse.Namespace) -> int:
    raw_text = fetch_published_google_doc(args.url)
    predictions = predict_from_text(
        raw_text,
        args.model,
        source=args.url,
        db_path=args.db,
        day_one=args.mode == "day-one",
    )
    for row in predictions:
        print(f"{row.player_name}: {row.formatted}")
    return 0


def command_start_bot(args: argparse.Namespace) -> int:
    import asyncio
    import logging
    from .bot.client import MafiaBot

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bot = MafiaBot(args.config)

    async def _run() -> None:
        try:
            await bot.start()
        except KeyboardInterrupt:
            print("\nStopping bot...")
            await bot.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nStopping bot...")
        return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser("mafia_framework")
    parser.add_argument("--db", default="data/mafia.db", help="Path to SQLite database file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="Create or update the database schema.").set_defaults(func=command_init_db)

    ingest = subparsers.add_parser("ingest", help="Ingest a raw Showdown log file into the database.")
    ingest.add_argument("file")
    ingest.set_defaults(func=command_ingest)

    ingest_url = subparsers.add_parser("ingest-url", help="Ingest a published Google Docs URL.")
    ingest_url.add_argument("url")
    ingest_url.set_defaults(func=command_ingest_url)

    inspect = subparsers.add_parser("inspect", help="Inspect an ingested game.")
    inspect.add_argument("--game-id", type=int)
    inspect.set_defaults(func=command_inspect)

    delete = subparsers.add_parser("delete-game", help="Delete a stored game.")
    delete.add_argument("--game-id", type=int, required=True)
    delete.set_defaults(func=command_delete_game)

    set_name = subparsers.add_parser("set-game-name", help="Set a human-readable name for a game.")
    set_name.add_argument("--game-id", type=int, required=True)
    set_name.add_argument("--name", required=True)
    set_name.set_defaults(func=command_set_game_name)

    set_role = subparsers.add_parser("set-role", help="Set a player's flip alignment.")
    set_role.add_argument("--game-id", type=int, required=True)
    set_role.add_argument("--player", required=True)
    set_role.add_argument("--alignment", required=True, choices=["town", "mafia", "neutral", "unknown"])
    set_role.set_defaults(func=command_set_role)

    set_alias = subparsers.add_parser("set-alias", help="Map a player alias to a canonical name.")
    set_alias.add_argument("--alias", required=True)
    set_alias.add_argument("--canonical", required=True)
    set_alias.set_defaults(func=command_set_alias)

    subparsers.add_parser("list-aliases", help="List player alias mappings.").set_defaults(func=command_list_aliases)

    remove_alias = subparsers.add_parser("remove-alias", help="Remove a player alias.")
    remove_alias.add_argument("--alias", required=True)
    remove_alias.set_defaults(func=command_remove_alias)

    import_aliases = subparsers.add_parser("import-aliases", help="Import aliases from JSON.")
    import_aliases.add_argument("file")
    import_aliases.set_defaults(func=command_import_aliases)

    predict = subparsers.add_parser("predict", help="Predict alignment from a raw game log.")
    predict.add_argument("--model", default="data/model.pkl")
    predict.add_argument("--mode", choices=["full", "day-one"], default="full")
    predict.add_argument("file")
    predict.set_defaults(func=command_predict)

    predict_url = subparsers.add_parser("predict-url", help="Predict from a published Google Docs URL.")
    predict_url.add_argument("--model", default="data/model.pkl")
    predict_url.add_argument("--mode", choices=["full", "day-one"], default="full")
    predict_url.add_argument("url")
    predict_url.set_defaults(func=command_predict_url)

    train = subparsers.add_parser("train", help="Train an alignment model.")
    train.add_argument("--output", default=None)
    train.add_argument("--mode", choices=["full", "day-one"], default="full")
    train.add_argument("--exclude-neutral-games", action="store_true")
    train.set_defaults(func=command_train)

    start_bot = subparsers.add_parser("start-bot", help="Start the live Pokemon Showdown Mafia bot.")
    start_bot.add_argument("--config", default="config.toml", help="Path to bot configuration file.")
    start_bot.add_argument("--debug", action="store_true", help="Enable debug logging (shows all websockets traffic).")
    start_bot.set_defaults(func=command_start_bot)

    argv: list[str] = []
    db_args: list[str] = []
    raw_args = list(sys.argv[1:])
    index = 0
    while index < len(raw_args):
        arg = raw_args[index]
        if arg == "--db":
            db_args.extend([arg, raw_args[index + 1]] if index + 1 < len(raw_args) else [arg])
            index += 2
            continue
        argv.append(arg)
        index += 1

    args = parser.parse_args(db_args + argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
