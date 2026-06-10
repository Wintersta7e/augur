"""Playable two-player chess board with timing data sent to Redis and NATS."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import chess
import nats
import pygame
import redis

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from tabula.config import AugurConfig
from tabula.connections import connect_redis
from tabula.contracts import PerceptionEvent
from tabula.session import SessionManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("chess_board")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SQUARE_SIZE = 80
BOARD_PX = SQUARE_SIZE * 8
INFO_HEIGHT = 48
WINDOW_SIZE = (BOARD_PX, BOARD_PX + INFO_HEIGHT)

COLOR_LIGHT = pygame.Color("#F0D9B5")
COLOR_DARK = pygame.Color("#B58863")
COLOR_HIGHLIGHT = pygame.Color(186, 202, 68, 160)
COLOR_LEGAL = pygame.Color(100, 180, 100, 120)
COLOR_BG = pygame.Color("#302E2B")
COLOR_TEXT = pygame.Color("#FFFFFF")
COLOR_CHECK = pygame.Color(235, 64, 52, 180)

PIECE_UNICODE: dict[str, str] = {
    "P": "\u2659",
    "N": "\u2658",
    "B": "\u2657",
    "R": "\u2656",
    "Q": "\u2655",
    "K": "\u2654",
    "p": "\u265f",
    "n": "\u265e",
    "b": "\u265d",
    "r": "\u265c",
    "q": "\u265b",
    "k": "\u265a",
}

REDIS_KEY_LAST = "augur:chess:last_move"
REDIS_KEY_HISTORY = "augur:chess:move_history"
REDIS_HISTORY_MAX = 20
NATS_SUBJECT = "augur.perception.chess"

# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------
#
# ARCH-11: the local connect_redis wrapper was deleted; callers now use
# the shared helper from tabula.connections (imported above).


def publish_move_redis(r: redis.Redis, payload: dict) -> None:
    raw = json.dumps(payload)
    r.set(REDIS_KEY_LAST, raw)
    r.lpush(REDIS_KEY_HISTORY, raw)
    r.ltrim(REDIS_KEY_HISTORY, 0, REDIS_HISTORY_MAX - 1)
    log.info("Redis: wrote move %s", payload["move_san"])


# ---------------------------------------------------------------------------
# NATS helpers (async, run from sync context via event loop)
# ---------------------------------------------------------------------------


class NatsPublisher:
    """Thin wrapper that keeps a NATS connection open for the session."""

    def __init__(self, config: AugurConfig) -> None:
        self._config = config
        self._nc: Optional[nats.NATS] = None
        self._loop = asyncio.new_event_loop()

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        self._loop.run_until_complete(self._async_connect())

    async def _async_connect(self) -> None:
        # ARCH-04: use AugurConfig instead of hardcoded localhost so Docker
        # deploy mode (where NATS runs as a named container) works.
        self._nc = await nats.connect(
            self._config.nats_url,
            connect_timeout=self._config.nats_connect_timeout,
        )
        log.info("NATS connected (%s)", self._config.nats_url)

    def close(self) -> None:
        if self._nc:
            self._loop.run_until_complete(self._nc.close())
            log.info("NATS closed")
        self._loop.close()

    # -- publish -------------------------------------------------------------
    def publish(self, subject: str, payload: dict) -> None:
        self._loop.run_until_complete(self._async_publish(subject, payload))

    async def _async_publish(self, subject: str, payload: dict) -> None:
        if self._nc is None:
            return
        await self._nc.publish(subject, json.dumps(payload).encode())
        log.info("NATS: published to %s", subject)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_board(
    surface: pygame.Surface, selected_sq: Optional[int], board: chess.Board
) -> None:
    """Draw the 8x8 board squares, highlights, and check indicator."""
    for sq in range(64):
        rank, file = divmod(sq, 8)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE
        is_light = (file + rank) % 2 == 0
        color = COLOR_LIGHT if is_light else COLOR_DARK
        pygame.draw.rect(surface, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

    # Highlight king in check
    if board.is_check():
        king_sq = board.king(board.turn)
        if king_sq is not None:
            file = chess.square_file(king_sq)
            rank = chess.square_rank(king_sq)
            check_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            check_surf.fill(COLOR_CHECK)
            surface.blit(check_surf, (file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE))

    # Highlight selected square and legal move targets
    if selected_sq is not None:
        file = chess.square_file(selected_sq)
        rank = chess.square_rank(selected_sq)
        sel_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        sel_surf.fill(COLOR_HIGHLIGHT)
        surface.blit(sel_surf, (file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE))

        for move in board.legal_moves:
            if move.from_square == selected_sq:
                tf = chess.square_file(move.to_square)
                tr = chess.square_rank(move.to_square)
                dot_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                radius = SQUARE_SIZE // 6
                pygame.draw.circle(
                    dot_surf,
                    COLOR_LEGAL,
                    (SQUARE_SIZE // 2, SQUARE_SIZE // 2),
                    radius,
                )
                surface.blit(dot_surf, (tf * SQUARE_SIZE, (7 - tr) * SQUARE_SIZE))


def draw_pieces(
    surface: pygame.Surface, board: chess.Board, font: pygame.font.Font
) -> None:
    """Draw Unicode chess pieces on the board."""
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        char = PIECE_UNICODE[piece.symbol()]
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        x = file * SQUARE_SIZE + SQUARE_SIZE // 2
        y = (7 - rank) * SQUARE_SIZE + SQUARE_SIZE // 2
        text = font.render(
            char,
            True,
            pygame.Color("black")
            if piece.color == chess.BLACK
            else pygame.Color("white"),
        )
        # Outline for contrast
        outline = font.render(
            char,
            True,
            pygame.Color("white")
            if piece.color == chess.BLACK
            else pygame.Color("black"),
        )
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            surface.blit(outline, outline.get_rect(center=(x + dx, y + dy)))
        surface.blit(text, text.get_rect(center=(x, y)))


def draw_info(
    surface: pygame.Surface,
    board: chess.Board,
    font: pygame.font.Font,
    think_start: float,
) -> None:
    """Draw the status bar below the board."""
    bar_rect = pygame.Rect(0, BOARD_PX, BOARD_PX, INFO_HEIGHT)
    pygame.draw.rect(surface, COLOR_BG, bar_rect)

    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        status = f"Checkmate - {winner} wins!"
    elif board.is_stalemate():
        status = "Stalemate - Draw"
    elif board.is_insufficient_material():
        status = "Draw - Insufficient material"
    elif board.can_claim_threefold_repetition():
        status = "Draw claimable - Threefold repetition"
    elif board.can_claim_fifty_moves():
        status = "Draw claimable - Fifty-move rule"
    else:
        turn = "White" if board.turn == chess.WHITE else "Black"
        elapsed = time.monotonic() - think_start
        check = " (CHECK)" if board.is_check() else ""
        status = f"{turn} to move{check}  |  Thinking: {elapsed:.1f}s"

    text = font.render(status, True, COLOR_TEXT)
    surface.blit(text, text.get_rect(midleft=(12, BOARD_PX + INFO_HEIGHT // 2)))


def sq_from_pixel(x: int, y: int) -> int:
    """Convert pixel coordinates to a chess square index."""
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    return chess.square(file, rank)


# ---------------------------------------------------------------------------
# Promotion dialog
# ---------------------------------------------------------------------------


def ask_promotion(
    surface: pygame.Surface, color: chess.Color, piece_font: pygame.font.Font
) -> chess.PieceType:
    """Show a simple 4-square overlay for promotion selection."""
    choices = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    symbols = {
        chess.QUEEN: "Q",
        chess.ROOK: "R",
        chess.BISHOP: "B",
        chess.KNIGHT: "N",
    }
    overlay = pygame.Surface((SQUARE_SIZE * 4, SQUARE_SIZE), pygame.SRCALPHA)
    overlay.fill(pygame.Color(50, 50, 50, 220))

    rects: list[pygame.Rect] = []
    ox = (BOARD_PX - SQUARE_SIZE * 4) // 2
    oy = BOARD_PX // 2 - SQUARE_SIZE // 2

    for i, pt in enumerate(choices):
        r = pygame.Rect(ox + i * SQUARE_SIZE, oy, SQUARE_SIZE, SQUARE_SIZE)
        rects.append(r)
        sym = symbols[pt].upper() if color == chess.WHITE else symbols[pt].lower()
        char = PIECE_UNICODE[sym if color == chess.WHITE else sym.lower()]
        txt = piece_font.render(char, True, COLOR_TEXT)
        pygame.draw.rect(surface, pygame.Color(80, 80, 80), r)
        pygame.draw.rect(surface, COLOR_TEXT, r, 2)
        surface.blit(txt, txt.get_rect(center=r.center))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for i, r in enumerate(rects):
                    if r.collidepoint(event.pos):
                        return choices[i]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Augur Chess")
    clock = pygame.time.Clock()

    piece_font = pygame.font.SysFont("segoeuisymbol,dejavusans,noto", SQUARE_SIZE - 12)
    info_font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 18)

    # Infrastructure connections (ARCH-04: routed through AugurConfig so
    # Docker deploy mode picks up the container network addresses).
    config = AugurConfig.from_env()

    try:
        redis_client = connect_redis(config)
    except redis.ConnectionError as exc:
        log.error("Cannot connect to Redis: %s", exc)
        sys.exit(1)

    nats_pub = NatsPublisher(config)
    try:
        nats_pub.connect()
    except Exception as exc:
        log.error("Cannot connect to NATS: %s", exc)
        sys.exit(1)

    # Session management
    session_mgr = SessionManager(redis_client)
    session_id = session_mgr.start()

    # Publish session start to NATS
    try:
        nats_pub.publish(
            "augur.session.start",
            {
                "session_id": session_id,
                "domain": "chess",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:
        log.error("Failed to publish session start: %s", exc)

    board = chess.Board()
    selected_sq: Optional[int] = None
    think_start = time.monotonic()
    move_number = 1
    running = True

    while running:
        # -- Events ----------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset board
                    board.reset()
                    selected_sq = None
                    think_start = time.monotonic()
                    move_number = 1
                    log.info("Board reset")
                elif event.key == pygame.K_u:
                    # Undo last move
                    if board.move_stack:
                        board.pop()
                        selected_sq = None
                        think_start = time.monotonic()
                        move_number = max(1, (board.fullmove_number))
                        log.info("Undo")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if board.is_game_over():
                    continue
                mx, my = event.pos
                if my >= BOARD_PX:
                    continue
                clicked_sq = sq_from_pixel(mx, my)

                if selected_sq is None:
                    # Select a piece of the current player
                    piece = board.piece_at(clicked_sq)
                    if piece and piece.color == board.turn:
                        selected_sq = clicked_sq
                else:
                    # Attempt move
                    piece = board.piece_at(selected_sq)

                    # Check for promotion
                    promotion = None
                    if piece and piece.piece_type == chess.PAWN:
                        target_rank = chess.square_rank(clicked_sq)
                        if (piece.color == chess.WHITE and target_rank == 7) or (
                            piece.color == chess.BLACK and target_rank == 0
                        ):
                            # Only ask if the move is potentially legal
                            test_move = chess.Move(
                                selected_sq, clicked_sq, promotion=chess.QUEEN
                            )
                            if test_move in board.legal_moves:
                                promotion = ask_promotion(
                                    screen, board.turn, piece_font
                                )

                    move = chess.Move(selected_sq, clicked_sq, promotion=promotion)

                    if move in board.legal_moves:
                        think_time = time.monotonic() - think_start
                        player = "white" if board.turn == chess.WHITE else "black"
                        san = board.san(move)
                        ts = datetime.now(timezone.utc).isoformat()

                        board.push(move)

                        # Redis payload (unchanged legacy format)
                        redis_payload = {
                            "player": player,
                            "move_uci": move.uci(),
                            "move_san": san,
                            "think_time_seconds": round(think_time, 3),
                            "move_number": move_number,
                            "timestamp": ts,
                        }

                        try:
                            publish_move_redis(redis_client, redis_payload)
                        except redis.RedisError as exc:
                            log.error("Redis publish failed: %s", exc)

                        # NATS payload (PerceptionEvent envelope)
                        event = PerceptionEvent(
                            domain="chess",
                            stream_id="chess_timing",
                            entity=player,
                            event_type="move",
                            value=round(think_time, 3),
                            unit="seconds",
                            context={
                                "move_uci": move.uci(),
                                "move_san": san,
                                "move_number": move_number,
                            },
                            timestamp=ts,
                            session_id=session_id,
                        )

                        try:
                            nats_pub.publish(NATS_SUBJECT, json.loads(event.to_json()))
                        except Exception as exc:
                            log.error("NATS publish failed: %s", exc)

                        if player == "black":
                            move_number += 1

                        think_start = time.monotonic()
                        selected_sq = None
                    else:
                        # Clicked on own piece -> re-select; otherwise deselect
                        new_piece = board.piece_at(clicked_sq)
                        if new_piece and new_piece.color == board.turn:
                            selected_sq = clicked_sq
                        else:
                            selected_sq = None

        # -- Draw ------------------------------------------------------------
        draw_board(screen, selected_sq, board)
        draw_pieces(screen, board, piece_font)
        draw_info(screen, board, info_font, think_start)
        pygame.display.flip()
        clock.tick(30)

    # Session end
    session_mgr.end()
    try:
        nats_pub.publish(
            "augur.session.end",
            {
                "session_id": session_id,
                "domain": "chess",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:
        log.error("Failed to publish session end: %s", exc)

    # Cleanup
    nats_pub.close()
    pygame.quit()


if __name__ == "__main__":
    main()
