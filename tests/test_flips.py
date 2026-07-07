import unittest

from mafia_framework.data.flips import FLIP_RE, extract_flips


class TestFlips(unittest.TestCase):

    def test_extract_flips(self):
        raw = """
        Alice's role was town
        Bob's role was mafia
        """

        flips = extract_flips(raw)
        self.assertEqual(len(flips), 2)
        self.assertEqual({flip.player_name for flip in flips}, {"Alice", "Bob"})
        self.assertEqual({flip.alignment for flip in flips}, {"town", "mafia"})

    def test_flip_regex_captures_player_and_role(self):
        raw = "Biwhohatesgliscorg's role was Vanilla Townie."
        match = FLIP_RE.search(raw)
        self.assertIsNotNone(match)
        self.assertEqual(match.group('player'), 'Biwhohatesgliscorg')
        self.assertEqual(match.group('role'), 'Vanilla Townie')

        debug_statement = (
            f"regex={FLIP_RE.pattern!r}; "
            f"player={match.group('player')!r}; "
            f"role={match.group('role')!r}"
        )

        expected = f"regex={repr(FLIP_RE.pattern)}; player='Biwhohatesgliscorg'; role='Vanilla Townie'"
        self.assertEqual(debug_statement, expected)

    def test_extract_flips_vanilla_townie_format(self):
        raw = """
        Crespo XY was eliminated!
        Crespo XY's role was Vanilla Townie.

        mist was eliminated!
        mist's role was Vanilla Townie.
        """

        flips = extract_flips(raw)
        self.assertEqual(len(flips), 2)
        self.assertEqual({flip.player_name for flip in flips}, {"Crespo XY", "mist"})
        self.assertEqual({flip.alignment for flip in flips}, {"town"})

    def test_extract_flips_handles_elimination_and_reveal_in_one_sentence(self):
        # Some sources bundle the elimination notice and the role reveal into
        # a single sentence with no line break between them. Without special
        # handling, the non-greedy player capture in FLIP_RE backtracks across
        # the whole "X was eliminated!" clause and mangles the player name.
        raw = "zorqbot was eliminated! zorqbot's role was Mafia Goon."

        flips = extract_flips(raw)
        self.assertEqual(len(flips), 1)
        self.assertEqual(flips[0].player_name, "zorqbot")
        self.assertEqual(flips[0].alignment, "mafia")
