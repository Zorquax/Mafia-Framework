from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Allow `streamlit run src/mafia_framework/dashboard/app.py` without editable install.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mafia_framework.data.aliases import import_aliases_from_json, list_player_aliases, remove_player_alias, set_player_alias
from mafia_framework.io.google_docs import fetch_published_google_doc
from mafia_framework.io.ingestion import ingest_google_doc, ingest_log
from mafia_framework.services.format_service import format_feature_value, format_percent
from mafia_framework.services.game_service import (
    assign_player_role,
    assign_player_roles_bulk,
    find_undefined_players,
    get_game_display_name,
    get_session,
    list_game_summaries,
    remove_game,
    resolve_flip_map,
    set_game_display_name,
)
from mafia_framework.services.game_log_service import build_game_log, parse_player_query
from mafia_framework.services.keyword_corpus import build_alignment_word_stats, player_alignment_word_usage
from mafia_framework.services.model_service import (
    get_feature_importance,
    get_player_tell_comparison,
    predict_from_text,
    predict_session,
    train_model_from_db,
    format_train_metrics,
)
from mafia_framework.services.tell_service import get_session_tells, list_canonical_players
from mafia_framework.analysis.tells import DAY_ONE_FEATURE_NAMES, FEATURE_NAMES
from mafia_framework.paths import resolve_repo_path


def _init_state() -> None:
    # Resolved eagerly (rather than left as a bare relative string) so the
    # dashboard finds the real database/models regardless of which
    # directory `streamlit run` was launched from.
    st.session_state.setdefault("db_path", str(resolve_repo_path("data/mafia.db")))
    st.session_state.setdefault("model_path", str(resolve_repo_path("data/model.pkl")))
    st.session_state.setdefault("model_d1_path", str(resolve_repo_path("data/model_d1.pkl")))


def _sidebar_config() -> tuple[str, str, str]:
    st.sidebar.title("Mafia Framework")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Games",
            "Game Log",
            "Ingest",
            "Train",
            "Predict",
            "Analytics",
            "Day 1",
            "Aliases",
        ],
    )
    db_path = str(resolve_repo_path(st.sidebar.text_input("Database path", st.session_state["db_path"])))
    model_path = str(resolve_repo_path(st.sidebar.text_input("Full model path", st.session_state["model_path"])))
    model_d1_path = str(resolve_repo_path(st.sidebar.text_input("Day 1 model path", st.session_state["model_d1_path"])))
    st.session_state["db_path"] = db_path
    st.session_state["model_path"] = model_path
    st.session_state["model_d1_path"] = model_d1_path
    return page, db_path, model_path


def _truncate(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def page_games(db_path: str) -> None:
    st.header("Games")

    all_undefined = find_undefined_players(db_path)
    if all_undefined:
        with st.expander(f"All undefined players ({len(all_undefined)})", expanded=False):
            undefined_table = pd.DataFrame(
                [
                    {
                        "game_id": row.game_id,
                        "name": row.display_name or "",
                        "player": row.player_name,
                        "has_messages": row.has_messages,
                    }
                    for row in all_undefined
                ]
            )
            filter_id = st.number_input("Filter by game id (0 = all)", min_value=0, value=0, step=1)
            if filter_id > 0:
                undefined_table = undefined_table[undefined_table["game_id"] == filter_id]
            st.dataframe(undefined_table, use_container_width=True)

    summaries = list_game_summaries(db_path)
    if not summaries:
        st.info("No games ingested yet.")
        return

    table = pd.DataFrame(
        [
            {
                "id": summary.game_id,
                "name": summary.display_name or "",
                "source": _truncate(summary.source),
                "players": summary.player_count,
                "flips": summary.flip_count,
                "undefined": summary.undefined_count,
                "needs_review": summary.needs_review,
                "created_at": summary.created_at,
            }
            for summary in summaries
        ]
    )
    st.dataframe(table, use_container_width=True)

    game_id = st.number_input("Game id", min_value=1, value=int(summaries[-1].game_id), step=1)
    session = get_session(db_path, int(game_id))
    if session is None:
        st.warning("Game not found.")
        return

    st.text_input("Source URL/Path (Copyable)", value=session.source, disabled=True, key=f"source-{game_id}")

    col1, col2 = st.columns(2)
    with col1:
        current_name = get_game_display_name(db_path, int(game_id)) or ""
        new_name = st.text_input("Display name", value=current_name)
        if st.button("Save display name"):
            set_game_display_name(db_path, int(game_id), new_name)
            st.success("Saved display name.")
            st.rerun()
    with col2:
        if st.button("Delete game", type="secondary"):
            remove_game(db_path, int(game_id))
            st.warning(f"Deleted game {game_id}.")
            st.rerun()

    st.subheader("Flips")
    resolved = resolve_flip_map(session)
    flip_rows = [{"player": name, "alignment": alignment} for name, alignment in sorted(resolved.items())]
    st.dataframe(pd.DataFrame(flip_rows) if flip_rows else pd.DataFrame(columns=["player", "alignment"]))

    st.subheader("Undefined players")
    undefined = find_undefined_players(db_path, int(game_id))
    if undefined:
        bulk_assignments: list[tuple[int, str, str]] = []
        for row in undefined:
            cols = st.columns([2, 1, 1, 2])
            cols[0].write(row.player_name)
            cols[1].write("has chat" if row.has_messages else "silent")
            cols[2].write("inferred town?" if row.is_inferred_town_candidate else "")
            default_alignment = "town" if row.is_inferred_town_candidate else ""
            alignment = cols[3].selectbox(
                "Set role",
                ["", "town", "mafia", "neutral", "unknown"],
                index=(["", "town", "mafia", "neutral", "unknown"].index(default_alignment) if default_alignment else 0),
                key=f"role-{row.game_id}-{row.player_name}",
                label_visibility="collapsed",
            )
            if alignment:
                bulk_assignments.append((row.game_id, row.player_name, alignment))
            if alignment and cols[3].button("Apply", key=f"apply-{row.game_id}-{row.player_name}"):
                assign_player_role(db_path, row.game_id, row.player_name, alignment)
                st.success(f"Set {row.player_name} to {alignment}")
                st.rerun()

        if bulk_assignments and st.button("Apply all selected roles", type="primary"):
            applied = assign_player_roles_bulk(db_path, bulk_assignments)
            st.success(f"Applied {applied} role assignment(s).")
            st.rerun()
    else:
        st.success("No undefined players in this game.")

    with st.expander("Full tell features"):
        tells = get_session_tells(db_path, int(game_id))
        if tells.empty:
            st.write("No tell data.")
        else:
            display = tells.copy()
            for column in display.columns:
                if column == "player_name":
                    continue
                display[column] = display[column].apply(
                    lambda value, col=column: format_feature_value(col, float(value))
                )
            st.dataframe(display, use_container_width=True)


def page_game_log(db_path: str) -> None:
    st.header("Game Log")
    summaries = list_game_summaries(db_path)
    if not summaries:
        st.info("No games ingested.")
        return

    game_id = st.selectbox(
        "Game",
        options=[summary.game_id for summary in summaries],
        format_func=lambda gid: next(
            (f"{s.game_id} — {s.display_name or _truncate(s.source)}" for s in summaries if s.game_id == gid),
            str(gid),
        ),
        key="game-log-select",
    )
    session = get_session(db_path, int(game_id))
    if session is None:
        st.warning("Game not found.")
        return

    st.caption(f"Roster: {', '.join(session.players)}")
    players_input = st.text_input(
        "Players (comma-separated)",
        placeholder="commanderawesome, aziziller, zorquax",
    )
    mode_label = st.radio("Show", ["both", "lines only", "votes only"], horizontal=True)
    mode_map = {"both": "both", "lines only": "lines", "votes only": "votes"}
    mode = mode_map[mode_label]

    if st.button("Load log", type="primary") and players_input.strip():
        queries = parse_player_query(players_input)
        entries = build_game_log(session, queries, mode=mode)
        if not entries:
            st.warning("No matching players or log entries found.")
            return

        current_day: int | None = None
        for entry in entries:
            if entry.day != current_day:
                current_day = entry.day
                st.markdown(f"### Day {current_day}")
            if entry.entry_type == "phase":
                st.markdown(f"**{entry.text}**")
            elif entry.entry_type == "message":
                timestamp = entry.timestamp or "?"
                st.text(f"[{timestamp}] {entry.player_name}: {entry.text}")
            elif entry.entry_type == "elimination":
                timestamp_prefix = f"[{entry.timestamp}] " if entry.timestamp else ""
                st.markdown(f"💥 **{timestamp_prefix}{entry.text}**")
            elif entry.entry_type == "reveal":
                timestamp_prefix = f"[{entry.timestamp}] " if entry.timestamp else ""
                st.markdown(f"🔍 *{timestamp_prefix}{entry.text}*")
            else:
                timestamp = entry.timestamp or "?"
                st.text(f"[{timestamp}] {entry.text}")


def page_ingest(db_path: str) -> None:
    st.header("Ingest")
    uploaded = st.file_uploader("Upload log file", type=["txt", "md", "log"])
    url = st.text_input("Google Docs /pub URL")

    if uploaded and st.button("Ingest file"):
        raw_text = uploaded.read().decode("utf-8", errors="replace")
        game_id = ingest_log(db_path, raw_text, source=uploaded.name)
        st.success(f"Ingested as game id {game_id}. Set a display name on the Games page.")

    if url and st.button("Ingest URL"):
        game_id = ingest_google_doc(db_path, url)
        st.success(f"Ingested as game id {game_id}.")


def page_train(db_path: str, model_path: str) -> None:
    st.header("Train")
    exclude_neutral = st.checkbox("Exclude games with neutral flips")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Train full model", type="primary"):
            try:
                result = train_model_from_db(
                    db_path,
                    model_path,
                    day_one=False,
                    exclude_neutral_games=exclude_neutral,
                )
                metrics = format_train_metrics(result.bundle)
                st.success(f"Saved to {result.output_path}")
                st.json(metrics)
            except ValueError as exc:
                st.error(str(exc))
    with col2:
        d1_path = st.session_state["model_d1_path"]
        if st.button("Train Day 1 model"):
            try:
                result = train_model_from_db(
                    db_path,
                    d1_path,
                    day_one=True,
                    exclude_neutral_games=exclude_neutral,
                )
                metrics = format_train_metrics(result.bundle)
                st.success(f"Saved to {result.output_path}")
                st.json(metrics)
            except ValueError as exc:
                st.error(str(exc))


def page_predict(db_path: str, model_path: str) -> None:
    st.header("Predict")
    mode = st.radio("Model mode", ["full", "day-one"], horizontal=True)
    selected_model = model_path if mode == "full" else st.session_state["model_d1_path"]
    uploaded = st.file_uploader("Upload log", type=["txt", "md", "log"], key="predict-file")
    url = st.text_input("Or paste /pub URL", key="predict-url")

    predictions = None
    if uploaded and st.button("Run prediction on file"):
        raw_text = uploaded.read().decode("utf-8", errors="replace")
        predictions = predict_from_text(
            raw_text,
            selected_model,
            source=uploaded.name,
            db_path=db_path,
            day_one=mode == "day-one",
        )
    elif url and st.button("Run prediction on URL"):
        raw_text = fetch_published_google_doc(url)
        predictions = predict_from_text(
            raw_text,
            selected_model,
            source=url,
            db_path=db_path,
            day_one=mode == "day-one",
        )

    if predictions:
        rows = [
            {
                "player": row.player_name,
                "town": row.formatted.get("town", "0.00%"),
                "mafia": row.formatted.get("mafia", "0.00%"),
                "top_signal": row.top_signal,
            }
            for row in predictions
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def page_analytics(db_path: str, model_path: str) -> None:
    st.header("Analytics")
    tab1, tab2, tab3 = st.tabs(["Player tells", "Alignment words", "Global feature importance"])

    with tab1:
        players = list_canonical_players(db_path)
        if not players:
            st.info("No players found.")
        else:
            player = st.selectbox("Player", players)
            day_one = st.checkbox("Use Day 1 features", value=False)
            comparison = get_player_tell_comparison(db_path, player, day_one=day_one)
            if comparison.empty:
                st.warning("No labeled games for this player.")
            else:
                top = comparison.head(12)
                fig = go.Figure()
                fig.add_bar(name="Player", x=top["feature"], y=top["player_mean"])
                fig.add_bar(name="Town avg", x=top["feature"], y=top["town_mean"])
                fig.add_bar(name="Mafia avg", x=top["feature"], y=top["mafia_mean"])
                fig.update_layout(barmode="group", title=f"Top tells for {player}")
                st.plotly_chart(fig, use_container_width=True)
                display = top.copy()
                for column in ["player_mean", "town_mean", "mafia_mean"]:
                    display[column] = display[column].apply(
                        lambda value, col=column: format_feature_value(col.replace("_mean", ""), float(value))
                        if "_mean" in column
                        else f"{value:.2f}"
                    )
                st.dataframe(display, use_container_width=True)

    with tab2:
        players = list_canonical_players(db_path)
        if not players:
            st.info("No players found.")
        else:
            player = st.selectbox("Player for word usage", players, key="alignment-word-player")
            usage = player_alignment_word_usage(db_path, player)
            if not usage:
                st.warning("No alignment-skewed words found for this player yet. Train corpus from more labeled games.")
            else:
                usage_df = pd.DataFrame(usage)
                st.dataframe(usage_df, use_container_width=True)
                st.caption("Words skew toward mafia/town based on corpus-wide alignment usage patterns.")

        if st.button("Show corpus alignment words"):
            stats = build_alignment_word_stats(db_path)
            if not stats:
                st.info("Not enough labeled chat data to derive alignment-specific words.")
            else:
                stats_df = pd.DataFrame(
                    [
                        {
                            "word": row.word,
                            "bias": row.alignment_bias,
                            "town_count": row.town_count,
                            "mafia_count": row.mafia_count,
                        }
                        for row in stats[:100]
                    ]
                )
                st.dataframe(stats_df, use_container_width=True)

    with tab3:
        if st.button("Load feature importance"):
            try:
                importance = get_feature_importance(model_path)
                fig = px.bar(
                    importance,
                    x="abs_coefficient",
                    y="feature",
                    color="direction",
                    orientation="h",
                    title="Logistic regression coefficients (|coef|)",
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))


def page_day_one(db_path: str) -> None:
    st.header("Day 1 view")
    summaries = list_game_summaries(db_path)
    if not summaries:
        st.info("No games ingested.")
        return

    game_id = st.selectbox(
        "Game",
        options=[summary.game_id for summary in summaries],
        format_func=lambda gid: next(
            (f"{s.game_id} — {s.display_name or _truncate(s.source)}" for s in summaries if s.game_id == gid),
            str(gid),
        ),
    )
    tells = get_session_tells(db_path, int(game_id), day_one=True)
    if tells.empty:
        st.warning("No Day 1 tell data.")
        return

    feature_cols = [col for col in tells.columns if col != "player_name"]
    heatmap = tells.set_index("player_name")[feature_cols]
    fig = px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Day 1 tell heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

    display = tells.copy()
    for column in feature_cols:
        display[column] = display[column].apply(lambda value, col=column: format_feature_value(col, float(value)))
    st.dataframe(display, use_container_width=True)

    d1_model = st.session_state["model_d1_path"]
    session = get_session(db_path, int(game_id))
    if session and st.button("Run Day 1 predictions"):
        try:
            predictions = predict_session(session, d1_model, day_one=True, db_path=db_path)
            rows = [
                {
                    "player": row.player_name,
                    "town": row.formatted.get("town", "0.00%"),
                    "mafia": row.formatted.get("mafia", "0.00%"),
                }
                for row in predictions
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))


def page_aliases(db_path: str) -> None:
    st.header("Aliases")
    rows = list_player_aliases(db_path)
    st.dataframe(pd.DataFrame(rows, columns=["alias", "canonical"]) if rows else pd.DataFrame(columns=["alias", "canonical"]))

    col1, col2 = st.columns(2)
    with col1:
        alias = st.text_input("Alias")
        canonical = st.text_input("Canonical name")
        if st.button("Add alias") and alias and canonical:
            set_player_alias(db_path, alias, canonical)
            st.success("Alias saved.")
            st.rerun()
    with col2:
        remove = st.text_input("Alias to remove")
        if st.button("Remove alias") and remove:
            remove_player_alias(db_path, remove)
            st.rerun()

    uploaded = st.file_uploader("Import aliases JSON", type=["json"])
    if uploaded and st.button("Import JSON"):
        path = f"/tmp/{uploaded.name}"
        with open(path, "wb") as handle:
            handle.write(uploaded.getbuffer())
        count = import_aliases_from_json(db_path, path)
        st.success(f"Imported {count} aliases.")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Mafia Framework", layout="wide")
    _init_state()
    page, db_path, model_path = _sidebar_config()

    if page == "Games":
        page_games(db_path)
    elif page == "Game Log":
        page_game_log(db_path)
    elif page == "Ingest":
        page_ingest(db_path)
    elif page == "Train":
        page_train(db_path, model_path)
    elif page == "Predict":
        page_predict(db_path, model_path)
    elif page == "Analytics":
        page_analytics(db_path, model_path)
    elif page == "Day 1":
        page_day_one(db_path)
    elif page == "Aliases":
        page_aliases(db_path)


if __name__ == "__main__":
    main()
